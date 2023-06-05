# Deep Latent Variable Models for Semi-supervised Paraphrase Generation

Code for AI Open 2023 journal: [Language as a latent sequence: deep latent variable models for semi-supervised paraphrase generation](https://www.sciencedirect.com/science/article/pii/S2666651023000025)

The main code is in "src/trainer.py"

Any files with ".sh" in src folder is a demonstration of experimental setup in this paper.

If any script with "--use_lm=True", you are expected to run the script named "run_lm.sh" first.

Be careful with the "--data" flag as currently all model saving/loading directories are shared, to avoid any problem you might want to change your setting in "src/config.py".

The "requirements.txt" is provided as a hint on the version of packages used in this project.
