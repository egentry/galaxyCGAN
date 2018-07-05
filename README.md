# GalaxyCGAN
A Generative Adversarial Network that generates images of galaxies for an input redshift and stellar mass

![example GAN results](https://raw.githubusercontent.com/egentry/galaxyCGAN/master/images/gan_example_for_readme.png)

# Where to start
## Data
Data is stored in the `data` dir. In there you'll find:
 - `HSC_ids.npy` - a list of object IDs for cross-referencing between files
 - `images.small.npy` - an array of galaxy images, of the shape `[n_ids, n_channels=3, x_pix=50, y_pix=50]`. The images are ordered to match that of `HSC_ids.npy`. (I.e., the nth image in `images.small.npy` has an id of the nth value in `HSC_ids.npy`.)
 - `2018_02_23-all_objects.csv` - a table of galaxy properties, of which `HSC_id` is a _non-unique_ key. The table is designed for COSMOS_id to be a primary key, and HSC_id to be a foreign key. As such multiple COSMOS-identify galaxies might be pointing to a single HSC-identified galaxy.  This can give multiple sets of [COSMOS-measured] galaxy properties for a particular HSC-identified object. In practice I only retain one set of measurements, without giving any thought as to which one to keep.

## Code
Most of the interesting stuff is stored in notebooks. The filenames are pretty straightforward, but here's some extra help in understanding them:
 - `simple classifier.ipynb` : the baseline classifier, which only uses traditional data augmentation (reflections, translations, etc.)
 - `simple gan.ipynb` : training a simple conditional GAN to accept redshift and stellar mass, and output a "realistic" looking galaxy image
 - `classifier with DAGAN.ipynb` : uses the GAN from `simple gan.ipynb` to create training images for a classifier with the same architecture as `simple classifier.ipynb`. Trains the classifier on generated images; validates the results on real images.
 
That should be enough to do everything locally, but if you want to skip training the gan (~100 cpu hours), you can download the checkpoint files from [this directory on dropbox](https://www.dropbox.com/sh/izks7nrxqozx2i1/AABDljzyE1Y3W2c9r1_Vtv1Ya?dl=0). Note: I make no guarantee about these files being available after March 2018; after that point you should just train the gan yourself using `simple gan.ipynb`.

# Results
I have a [written presentation](https://docs.google.com/document/d/1b_hhbJe1BeYTPs-bdPF380XTENkypS9nfCQ3_uJaBXo/edit?usp=sharing), and a [presentation slides](https://docs.google.com/presentation/d/1xjLsMAiMoyPivnso6r9iNnvczVzGpb6Xe3NvX4osrRk/edit?usp=sharing) overviewing my results.

In short, we were able to create relatively good galaxy images with realistic-looking noise (see the image at the top of this README). Unfortunately, the galaxy images didn't always scale in intuitive / accurate ways with respect to the input variable they were being conditioned on. For instance, if you added more stars to a galaxy (specifically increasing stellar mass), it would sometimes get fainter, which is not what we wanted. We have some ideas of how to try to address this by further tweaking the loss function, but haven't gotten a chance to explore those options yet.

![example of gan-generated images not scaling correctly with conditional input](https://raw.githubusercontent.com/egentry/galaxyCGAN/master/images/gan_bad_scaling_example.png)
