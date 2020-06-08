# Training of summarization models

Models are trained on Multi-en-Wiki-News by default but this can be changed by changing the training parameters.

All the training scripts are in the folder ``scripts/`` and all the training parameters are in the folder ``args/``.

To use training parameters by default, just run:

```bash
cd path/to/MultiDocMultiLingualSum/
python train/run_training --model
```
where ``model`` is the model you want to train. Models available are:

- ``bert2bert``,
- ``bart``,
- ``bart_cnn``,
- ``t5``,
- ``t5_with_title``.

So to train ``bart``, you have to run: ``python train/run_training --bart``.

To add you model, create a ``json`` file in ``args/`` with your training parameters.
