import os

import wandb
import BatchMaker
import LearnTools
import Networks
import ImageTools
import argparse
import torch
import numpy as np
from tifffile import imwrite, imread
import matplotlib.pyplot as plt
from taufactor import metrics


def labels_from_biases(probs, biases):
    probs = np.asarray(probs)
    biases = np.asarray(biases, dtype=probs.dtype)
    bias_shape = (1, -1) + (1,) * (probs.ndim - 2)
    return np.argmax(probs + biases.reshape(bias_shape), axis=1)


def labels_from_probs(probs):
    return np.argmax(np.asarray(probs), axis=1)


def phase_fractions(labels, n_phases=3):
    return np.array([np.mean(labels == i) for i in range(n_phases)])


def fit_one_phase(probs, biases, phase, target):
    bias_shape = (1, -1) + (1,) * (probs.ndim - 2)
    shifted = probs + biases.reshape(bias_shape)

    phase_score = shifted[:, phase]
    other_scores = np.delete(shifted, phase, axis=1)
    other_max = np.max(other_scores, axis=1)

    margin = (other_max - phase_score).flatten()
    delta = np.quantile(margin, target)

    biases = biases.copy()
    biases[phase] += delta

    # remove arbitrary global offset; only relative biases matter
    biases -= biases[0]

    return biases


def iterative_fit_phase_biases(probs, targets, max_iter=20, tol=1e-4):
    targets = np.asarray(targets, dtype=float)
    targets = targets / targets.sum()

    biases = np.zeros(len(targets), dtype=float)
    best_biases = biases.copy()
    best_fracs = None
    best_err = np.inf

    for i in range(max_iter):
        for phase, target in enumerate(targets):
            biases = fit_one_phase(probs, biases, phase, target)

        labels = labels_from_biases(probs, biases)
        fracs = phase_fractions(labels, len(targets))
        err = np.max(np.abs(fracs - targets))

        if err < best_err:
            best_err = err
            best_biases = biases.copy()
            best_fracs = fracs.copy()

        print(i, "biases =", biases, "fractions =", fracs, "max_err =", err)

        if err <= tol:
            print(f"Converged after {i + 1} iterations")
            break

    return best_biases, best_fracs


def main(
        return_pfs=False,
        save_im=True,
        new_eval=False,
        phase_biases=None,
        calibrate_phase_biases=False,
        calibration_repeats=200,
        stitched_calibration_repeats=200,
        eval_mode=False,
        bn_batch_size=8,
        training_batch_size=8,
):
    # Parsing arguments:
    parser = argparse.ArgumentParser()

    args = LearnTools.return_args(parser)

    progress_dir, wd, wg = args.directory, args.widthD, args.widthG
    n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
    num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
    squash, down_sample = args.squash_phases, args.down_sample
    D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor
    size_to_evaluate, separator = args.volume_size_to_evaluate, args.separator
    g_file_name, super_sample = args.g_image_path, args.super_sampling
    phases_to_low, g_epoch_id = args.phases_low_res_idx, args.g_epoch_id

    # progress directory: allow absolute, otherwise keep old "progress/<name>"
    progress_main_dir = (
        progress_dir
        if os.path.isabs(progress_dir)
        else os.path.join("progress", progress_dir)
    )

    # weights path: join safely; keep old naming
    path_to_g_weights = os.path.join(progress_main_dir, f"g_weights{g_epoch_id}.pth")

    # image path: allow absolute, otherwise keep old "data/<name>"
    G_image_path = (
        g_file_name
        if os.path.isabs(g_file_name)
        else os.path.join("data", g_file_name)
    )

    rand_id = str(np.random.randint(10000))

    file_name = 'generated_tif' + rand_id + '.tif'
    random_sample_3D = True
    input_with_noise = True
    all_pore_input = False

    # crop the edges
    crop = 4

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print('device is ' + str(device))

    # the material indices to low-res:
    to_low_idx = torch.LongTensor(phases_to_low).to(device)

    # Number of channels in the training images. For color images this is 3
    if squash:
        if input_with_noise:
            nc_g = 3
        else:
            nc_g = 2
    else:
        if input_with_noise:
            nc_g = 1 + to_low_idx.size()[0] + 1 # channel for pore plus number of
        # material phases to low res.
        else:
            nc_g = 1 + to_low_idx.size()[0]

    # TODO make this more general, to support any number of discriminator phases
    if separator:
        nc_d = 2
    else:
        nc_d = 3  # three phases for the discriminator input

    G_net = Networks.generator(ngpu, wg, nc_g, nc_d, n_res_blocks, n_dims,
                            scale_factor=scale_f).to(device)
    G_net.load_state_dict(torch.load(path_to_g_weights, map_location=torch.device(
        device)))

    # If the whole network is saved:
    # G_net = torch.load(path_to_g_weights, map_location=torch.device(device))
    if not new_eval:
        G_net.eval()
    elif eval_mode:
        G_net.eval()
    else:
        G_net.train()

    def random_sample(im_3d, size_to_evaluate):
        if not new_eval:
            print(f'im 3d shape: {im_3d.shape}')
        start = np.random.randint(0, np.array(im_3d.shape[-3:]) - size_to_evaluate + 1)
        if not new_eval:
            print(f'start: {start}')
        end = start + size_to_evaluate
        return im_3d[..., start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    def add_noise_channel(crop):
        noise = torch.randn(
            crop.size(0),
            1,
            *crop.shape[2:],
            device=device,
            dtype=crop.dtype,
        )
        return torch.cat((crop, noise), dim=1)

    def collect_training_phase_fractions(BM_G, G_net, random_batch_size=8,
                                         n_repeats=200):
        vf_pore = []
        vf_am = []
        vf_cbd = []
        with torch.no_grad():
            for repeat in range(n_repeats):
                crop = BM_G.random_batch_for_fake(random_batch_size, 0)
                crop = add_noise_channel(crop)
                _, out = G_net(crop)
                labels = labels_from_probs(out.detach().cpu().numpy())
                vf_pore.append(np.mean(labels == 0))
                vf_am.append(np.mean(labels == 1))
                vf_cbd.append(np.mean(labels == 2))
                if repeat % 20 == 0:
                    print(f"Training-fraction repeat {repeat + 1}/{n_repeats} completed.")
                if repeat %100 == 0:
                    print(f"Current mean fractions: pore={np.mean(vf_pore):.4f}, am={np.mean(vf_am):.4f}, cbd={np.mean(vf_cbd):.4f}")
        return np.array([np.mean(vf_pore), np.mean(vf_am), np.mean(vf_cbd)])

    def collect_stitched_tile_probs(BM_G, G_net, output_size=128,
                                    random_batch_size=8, n_repeats=200):
        added_up_probs = None
        with torch.no_grad():
            for repeat in range(n_repeats):
                im_3d = BM_G.all_image_batch()
                low_res_size = int(output_size / scale_f)
                crop = random_sample(im_3d, np.array([low_res_size] * 3))
                crop = add_bn_context_batch(crop, im_3d, random_batch_size)
                crop = add_noise_channel(crop)
                out, _ = G_net(crop)
                quarter = int(out.size(-1) / 4)
                out = out[:1, :, quarter:-quarter, quarter:-quarter, quarter:-quarter]
                probs = out.detach().cpu().numpy()
                if added_up_probs is None:
                    added_up_probs = probs
                else:
                    added_up_probs = np.concatenate((added_up_probs, probs), axis=0)
                if repeat % 20 == 0:
                    print(f"Stitched-tile repeat {repeat + 1}/{n_repeats} completed.")
        return added_up_probs

    def batchnorm_state(G_net):
        state = []
        for module in G_net.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                state.append((
                    module,
                    module.running_mean.detach().clone(),
                    module.running_var.detach().clone(),
                    module.num_batches_tracked.detach().clone(),
                ))
        return state

    def restore_batchnorm_state(state):
        for module, running_mean, running_var, num_batches_tracked in state:
            module.running_mean.copy_(running_mean)
            module.running_var.copy_(running_var)
            module.num_batches_tracked.copy_(num_batches_tracked)

    def calibrate_stitched_phase_biases(BM_G, G_net, output_size=128,
                                        target_batch_size=8,
                                        stitched_batch_size=8):
        print("\n===== PHASE BIAS CALIBRATION =====")
        bn_state = batchnorm_state(G_net)
        G_net.train()
        phase_targets = collect_training_phase_fractions(
            BM_G, G_net, random_batch_size=target_batch_size,
            n_repeats=calibration_repeats,
        )
        restore_batchnorm_state(bn_state)
        if eval_mode:
            G_net.eval()
        else:
            G_net.train()
        stitched_probs = collect_stitched_tile_probs(
            BM_G, G_net, output_size=output_size,
            random_batch_size=stitched_batch_size,
            n_repeats=stitched_calibration_repeats,
        )
        unadjusted = phase_fractions(labels_from_probs(stitched_probs))
        biases, fitted_fracs = iterative_fit_phase_biases(
            stitched_probs, phase_targets, max_iter=20, tol=1e-3,
        )
        print(f"Training-like target fractions: {phase_targets}")
        print(f"Unadjusted stitched-tile fractions: {unadjusted}")
        print(f"Phase biases: {biases}")
        print(f"Bias-fitted stitched-tile fractions: {fitted_fracs}")
        return {
            "biases": biases,
            "target_fractions": phase_targets,
            "unadjusted_stitched_fractions": unadjusted,
            "fitted_stitched_fractions": fitted_fracs,
        }

    def run_wandb_test(BM_G, G_net, device, output_size=None, random_batch_size=32, n_repeats=100):
        print("\n===== WANDB REPRODUCTION TEST =====")
        vf_pore = []
        vf_am = []
        vf_cbd = []
        added_up_image = None
        added_up_probs = None
        with torch.no_grad():
            for repeat in range(n_repeats):
                # exactly the same type of input as training
                if output_size is None:
                    crop = BM_G.random_batch_for_fake(random_batch_size, 0)
                else:
                    im_3d = BM_G.all_image_batch()
                    low_res_size = int(output_size / scale_f)
                    crop = random_sample(
                        im_3d,
                        np.array([low_res_size] * 3)
                    )
                    crop = add_bn_context_batch(crop, im_3d, random_batch_size)
                crop = add_noise_channel(crop)
                # exactly the same output used in WandB
                out, _ = G_net(crop)
                quarter = int(out.size(-1) / 4)
                out = out[..., quarter:-quarter, quarter:-quarter, quarter:-quarter]
                # print(f"out shape: {out.shape}")
                probs = out.cpu()
                out = ImageTools.fractions_to_ohe(probs)
                out = ImageTools.one_hot_decoding(out)
                if added_up_image is None:
                    added_up_image = out
                    added_up_probs = probs.numpy()
                else:
                    added_up_image = np.concatenate((added_up_image, out), axis=0)
                    added_up_probs = np.concatenate((added_up_probs, probs.numpy()), axis=0)
                    print(f"added_up_probs shape: {added_up_probs.shape}")
                vf_pore.append(np.mean(out == 0))
                vf_am.append(np.mean(out == 1))
                vf_cbd.append(np.mean(out == 2))
                if repeat % 20 == 0:
                    print(f"Repeat {repeat + 1}/{n_repeats} completed.")
        print(f"Pore volume fraction: {np.mean(vf_pore):.4f} ± {np.std(vf_pore):.4f}")
        print(f"AM volume fraction:   {np.mean(vf_am):.4f} ± {np.std(vf_am):.4f}")
        print(f"CBD volume fraction:  {np.mean(vf_cbd):.4f} ± {np.std(vf_cbd):.4f}")
        print("===================================\n")
        # print(f"Added up probs shape: {added_up_probs.shape}")
        pore_target = 0.39377
        cbd_target = 0.1966
        am_target = 1 - pore_target - cbd_target
        phase_targets = np.array([pore_target, am_target, cbd_target])

        biases, fitted_fracs = iterative_fit_phase_biases(
            added_up_probs,
            phase_targets,
            max_iter=20,
            tol=1e-3,
        )

        new_out = labels_from_biases(added_up_probs, biases)

        print(f"New pore volume fraction: {np.mean(new_out == 0):.4f}")
        print(f"New AM volume fraction:   {np.mean(new_out == 1):.4f}")
        print(f"New CBD volume fraction:  {np.mean(new_out == 2):.4f}")

        # If you also need one-hot:
        plt.imshow(new_out[0,0], cmap='gray', vmin=0, vmax=2)
        plt.show()
        plt.imshow(added_up_image[0,0], cmap='gray', vmin=0, vmax=2)
        plt.show()

        # print(f"fraction of uncertain pore probes: {np.mean(uncertain_mask):.4f}")
        # plt.hist(pore_delta_needed, bins=50)
        # plt.yscale('log')
        # plt.show()
        return vf_pore, vf_am, vf_cbd

    def crop_to_down_sample(high_res):
        """
        If down sample, crops the high resolution image to fit the scale factor.
        """
        dims = np.array(high_res.shape)
        crop_dims = []
        for idx in range(len(dims)):
            dim = dims[idx]
            for subtract in range(dim):
                # doing % twice because the number can be 0 from below (%1.6=1.599)
                if np.round((dim - subtract) % scale_f, 5) % scale_f == 0:
                    crop_dims.append(dim - subtract)
                    break
        return high_res[:crop_dims[0], :crop_dims[1], :crop_dims[2]]

    def add_bn_context_batch(true_input, full_input_volume, n_bn_samples=32):
        """
        Create a batch for train-mode BatchNorm evaluation."""
        if n_bn_samples <= 1:
            return true_input

        crop_shape = true_input.shape[-3:]
        batch = [true_input]
        for _ in range(n_bn_samples - 1):
            batch.append(random_sample(full_input_volume, crop_shape))

        return torch.cat(batch, dim=0)

    with torch.no_grad():  # save the images
        # 1. Start a new run
        with_wandb = False
        if with_wandb:
            wandb.init(project='SuperRes', name='making large volume',
                   entity='tldr-group')
        super_res_output = 128
        step_len = int(np.round(super_res_output/scale_f, 5))
        overlap = int(step_len/2)
        high_overlap = int(np.round(overlap / 2 * scale_f, 5))
        step = step_len - overlap
        middle_output = super_res_output - 2 * high_overlap
        if new_eval:
            # The new evaluation keeps only the middle of every generated tile,
            # so use a crop size that is exactly covered by whole stitching steps.
            tile_counts = (
                np.ceil((np.array(size_to_evaluate) - step_len) / step).astype(int)
                + 1
            )
            stitch_size_to_evaluate = step_len + (tile_counts - 1) * step
        else:
            # The old evaluation keeps the outer edges of the boundary tiles and
            # handles a final partial step when the requested size is not aligned.
            stitch_size_to_evaluate = np.array(size_to_evaluate)

        print(f"step_len = {step_len}, overlap = {overlap}, high_overlap = {high_overlap}, step = {step}")

        BM_G = BatchMaker.\
            BatchMaker(device=device, to_low_idx=to_low_idx, path=G_image_path,
                    sf=scale_f, dims=n_dims, stack=False,
                    down_sample=down_sample, low_res=not down_sample,
                    rot_and_mir=False, squash=squash, super_sample=super_sample)
        im_3d = BM_G.all_image_batch()

        if not new_eval:
            G_net.eval()
        elif eval_mode:
            G_net.eval()
        else:
            G_net.train()  # important!
        if calibrate_phase_biases:
            return calibrate_stitched_phase_biases(
                BM_G, G_net, output_size=super_res_output,
                target_batch_size=training_batch_size,
                stitched_batch_size=bn_batch_size,
            )
        if return_pfs:
            return run_wandb_test(BM_G, G_net, device, random_batch_size=bn_batch_size, n_repeats=60)
        # run_wandb_test(BM_G, G_net, device, output_size=super_res_output, random_batch_size=8, n_repeats=20)
        # G_net.eval()  # important!
        # run_wandb_test(BM_G, G_net, device, random_batch_size=1)
        # return
        if all_pore_input:
            im_3d[:] = 0
            im_3d[:, 0] = 1

        if input_with_noise:
            input_size = im_3d.size()
            # make noise channel and concatenate it to input:
            noise = torch.randn(input_size[0], 1, *input_size[2:],
                                device=device, dtype=im_3d.dtype)
            im_3d = torch.cat((im_3d, noise), dim=1)
        
        if random_sample_3D:
            im_3d = random_sample(im_3d, stitch_size_to_evaluate)

        nz1, nz2, nz3 = stitch_size_to_evaluate
        first_img_stack = []
        with torch.no_grad():
            last_ind1 = int(np.ceil((nz1-step_len)/step))
            for i in range(last_ind1 + 1):
                print('large step = ' + str(i))
                if i == last_ind1:
                    first_lr_vec = im_3d[..., nz1-step_len:nz1, :, :]
                else:
                    first_lr_vec = im_3d[..., i*step:i*step+step_len, :, :]
                second_img_stack = []
                last_ind2 = int(np.ceil((nz2-step_len)/step))
                for j in range(last_ind2 + 1):
                    if j == last_ind2:
                        second_lr_vec = first_lr_vec[..., :, nz2-step_len:nz2, :]
                    else:
                        second_lr_vec = first_lr_vec[..., :, j * step:j * step +
                                                    step_len, :]
                    third_img_stack = []
                    last_ind3 = int(np.ceil((nz3-step_len)/step))
                    for k in range(last_ind3 + 1):
                        if k == last_ind3:
                            third_lr_vec = second_lr_vec[..., :, :,
                                        nz3-step_len:nz3]
                        else:
                            third_lr_vec = second_lr_vec[..., :, :, k * step:k *
                                                        step + step_len]
                        if new_eval:
                            third_lr_vec_for_G = add_bn_context_batch(
                                third_lr_vec,
                                im_3d,
                                n_bn_samples=bn_batch_size,
                            )
                            g_output, _ = G_net(third_lr_vec_for_G)
                            # Keep only the output corresponding to the true tile.
                            g_output = g_output[:1]
                            g_output = g_output.detach().cpu().numpy()
                            if phase_biases is None:
                                g_output_grey = labels_from_probs(g_output)
                            else:
                                g_output_grey = labels_from_biases(
                                    g_output, phase_biases
                                )
                            g_output_grey = g_output_grey.astype('int8').squeeze()
                            g_output_grey = g_output_grey[
                                high_overlap:-high_overlap,
                                high_overlap:-high_overlap,
                                high_overlap:-high_overlap,
                            ]
                        else:
                            # Exact generator call and decoding from the original
                            # Evaluation.py.
                            g_output, _ = G_net(third_lr_vec)
                            g_output = g_output.detach().cpu()
                            g_output = ImageTools.fractions_to_ohe(g_output)
                            g_output_grey = ImageTools.one_hot_decoding(
                                g_output
                            ).astype('int8').squeeze()
                            if k == 0:  # keep the beginning
                                g_output_grey = g_output_grey[
                                    :, :, :-high_overlap
                                ]
                            elif k == last_ind3:  # keep the middle+end
                                excess_voxels = int(
                                    ((nz3 - step_len) % step) * scale_f
                                )
                                if excess_voxels > 0:
                                    g_output_grey = g_output_grey[
                                        :, :, -(high_overlap + excess_voxels):
                                    ]
                                else:
                                    g_output_grey = g_output_grey[
                                        :, :, high_overlap:
                                    ]
                            else:  # keep the middle
                                g_output_grey = g_output_grey[
                                    :, :, high_overlap:-high_overlap
                                ]
                        third_img_stack.append(np.int8(g_output_grey))
                    res2 = np.concatenate(third_img_stack, axis=2)
                    if not new_eval:
                        if j == 0:
                            res2 = res2[:, :-high_overlap, :]
                        elif j == last_ind2:
                            excess_voxels = int(
                                ((nz2 - step_len) % step) * scale_f
                            )
                            if excess_voxels > 0:
                                res2 = res2[
                                    :, -(high_overlap + excess_voxels):, :
                                ]
                            else:
                                res2 = res2[:, high_overlap:, :]
                        else:
                            res2 = res2[:, high_overlap:-high_overlap, :]
                    second_img_stack.append(res2)
                res1 = np.concatenate(second_img_stack, axis=1)
                if not new_eval:
                    if i == 0:
                        res1 = res1[:-high_overlap, :, :]
                    elif i == last_ind1:
                        excess_voxels = int(
                            ((nz1 - step_len) % step) * scale_f
                        )
                        if excess_voxels > 0:
                            res1 = res1[
                                -(high_overlap + excess_voxels):, :, :
                            ]
                        else:
                            res1 = res1[high_overlap:, :, :]
                    else:
                        res1 = res1[high_overlap:-high_overlap, :, :]
                first_img_stack.append(res1)
        img = np.concatenate(first_img_stack, axis=0)
        if not new_eval:
            img = img[crop:-crop, crop:-crop, crop:-crop]
        if with_wandb:
            output_image = img[0]
            plt.imshow(output_image, cmap='gray', vmin=0, vmax=2)
            wandb.log({"generated image first slice": plt})
        low_res = np.squeeze(ImageTools.one_hot_decoding(im_3d.cpu())).astype('int8')
        if save_im:
            if all_pore_input:
                imwrite(progress_main_dir + '/' + file_name + '_pore', img)
            else:
                imwrite(progress_main_dir + '/' + file_name, img)

            # also save the low-res input.
            imwrite(progress_main_dir + '/' + file_name.split('.')[0] + '_low_res.tif',
                low_res)
        else:
            if new_eval:
                print(f"image output shape: {img.shape}, dtype: {img.dtype}")
            return img

if __name__ == "__main__":
    main()
