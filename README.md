# CapsVoxGAN

## Description

My project aims to be a three dimensional generative adversarial network (GAN) for generating voxel models, using a three dimensional [capsule network](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules). CNNs are quite good when it comes to detecting features, but they do not take spatial information into account. The following picture illustrates this:

![](https://miro.medium.com/max/383/1*b5Gj44sqne2Cu6Xra5EJpg.jpeg)

_Source:_ https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8

This picture looks somewhat like a face, but more like an abstract painting, than an actual face. All of the features required for a face are there, but their alignment is odd, so are their relative sizes. For two dimensional settings this is not optimal, but the results of CNNs are still good enough, so it is not a problem. In three dimensional settings, this is different: spatial arrangements are more important, due to the extra dimension, especially when generating models.

Capsule networks were introduced by Geoffrey Hinton, who is not pleased with the pooling operations in CNNs:
> The pooling operation used in convolutional neural networks is a big mistake and the fact that it works so well is a disaster. [_(Source)_](https://www.reddit.com/r/MachineLearning/comments/2lmo0l/ama_geoffrey_hinton/clyj4jv/)

They are a relatively new concept, but they have been used for [GANs](https://arxiv.org/abs/1802.06167) and for [3D data](https://arxiv.org/abs/1812.10775). To the best of my knowledge, this is the first attemt to use a GAN with a capsule network to generate a voxel model.

## Dataset

Machine learninging with 3D objects or scenes is a relatively new area, therefore there there is not as much annotated data available as for image recognition. Luckily in recent years this started to change, [here](https://github.com/timzhang642/3D-Machine-Learning#datasets) you can find a good overview of available datasets. 

The dataset of my choice is [ModelNet40](http://modelnet.cs.princeton.edu), it consits of 12311 models from 40 categories. The models are polygon meshes and therefore I have to convert them into voxel models first. I'll do this using [PyMesh](https://github.com/PyMesh/PyMesh), alternatively I could use [binvox](https://www.patrickmin.com/binvox/) together with [Gmsh](http://gmsh.info/). I was thinking of setting the grid of voxels to 64x64x64, as a compromise between computational effort and quality of the results, but this might be subject to change.

## Project Type

Bring your own method.

## Work-Breakdown:

Task | Hours
--- | ---
Getting familiar with the data / used libraries | 10
In-depth reading of related publications | 10
Coding of solution | 25
Creating presentation of results | 10

## References

### Papers

* [Sabour, Sara, Nicholas Frosst, and Geoffrey E. Hinton. "Dynamic routing between capsules." _Advances in neural information processing systems._ 2017.](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules)
* [Jaiswal, Ayush, et al. "Capsulegan: Generative adversarial capsule network." _Proceedings of the European Conference on Computer Vision (ECCV)._ 2018.](http://openaccess.thecvf.com/content_eccv_2018_workshops/w17/html/Jaiswal_CapsuleGAN_Generative_Adversarial_Capsule_Network_ECCVW_2018_paper.html)
* [Cheraghian, Ali, and Lars Petersson. "3DCapsule: Extending the Capsule Architecture to Classify 3D Point Clouds." _2019 IEEE Winter Conference on Applications of Computer Vision (WACV)._ IEEE, 2019.](https://ieeexplore.ieee.org/abstract/document/8658405)
* [Wu, Jiajun, et al. "Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling." _Advances in neural information processing systems._ 2016.](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling)
* [Brock, Andrew, et al. "Generative and discriminative voxel modeling with convolutional neural networks." _arXiv preprint arXiv:1608.04236_ (2016).](https://arxiv.org/abs/1608.04236)

### Datasets
* [ModelNet40](http://modelnet.cs.princeton.edu)

### GitHub Repositories

* [PyTorch implementation of capsule networks](https://github.com/gram-ai/capsule-networks) by @gram-ai
* [TensorFlow implementation of a GAN using capsule networks](https://github.com/gusgad/capsule-GAN) by @gusgad
* [PyTorch implementation of 3D Point Capsule Networks](https://github.com/yongheng1991/3D-point-capsule-networks) by @yongheng1991
* [Theano implementation of Voxel-Based Variational Autoencoders](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling) by @ajbrock
* [General resources about 3D machine learining](https://github.com/timzhang642/3D-Machine-Learning) by @timzhang642: 

### Libraries

* [PyMesh](https://pymesh.readthedocs.io/en/latest/)

### Tools

* [binvox](https://www.patrickmin.com/binvox/)
* [Gmsh](http://gmsh.info/)

###### N.B.

I am aware, that I might bite off more than I can chew, but whatever the final result will be, the journey is its own reward :smiley:
