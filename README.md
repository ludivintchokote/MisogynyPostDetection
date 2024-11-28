# Robust-hatememe-detection

This is the official repo of our paper entitled 

### HateProof: Are Hateful Meme Detection Systems really Robust?

Requirements:

To replicate the experiments, two separate environments are required.

For model ROBERTA+RESNET and Contrastive Learning models, following command can be used to create the environment:

```conda create --name <env> --file environments/requirements_fusion.txt```

For VISUALBERT and UNITER, please use following command:

```conda create --name <env> --file environments/requirements_prefused.txt```

For VILLA model, Docker container need to be created.


TBD

How to Contribute
If you have proposed a new robustness evaluation system, please fork this repository and raise pull request and then you can add it into this framework to evaluate. If there are any comments or problems, feel free to open an issue and ask us.
