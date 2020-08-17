This is the official code of Few-Shot Generalization for Single-Image 3D Reconstruction via Priors (ICCV 2019).

ShapeNet rendered images and voxels as used in this work are available for download at the [R2N2 Github](https://github.com/chrischoy/3D-R2N2#datasets).
Once downloaded, change the paths in L14/15 of `highres_sampler.py`.

The main training script is `train_iterative_RGB_refiner.py`. The most important argument is `--excluded-cats` which dictates which categories to hold out for few-shot learning.
Most experiments presented in our paper have `--excluded-cats benches,cabinets,lamps,sofas,vessels,rifles`.

Other scripts are remnants of various explorations/evaluations many of which didn't make it into the main paper. Feel free to explore/experiment with these and let me know if you find anything interesting!

Also please let me know if you have any questions concerning this code or the paper itself.
