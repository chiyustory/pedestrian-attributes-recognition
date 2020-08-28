# pedestrian-attributes-recognition

## Introdution

A [pytorch](https://github.com/pytorch/pytorch) implemented for pedestrian attributes recognition base. 

## Directory

- ### ```dataset```

  The data set file, which contains image path and labels of attributes.

  ```
  test.png 1 0 1
  ```

- ### ```model```

  The model of pre-trained or want to store.

- ### ```image```

  The image files.

- ### ```output```

  The log file in train or test process.

- ### ```src```

  - ```data```:  load data.
  - ```options```: parameters.
  - ```loss```: loss functions.
  - ```models```: networks.
  - ```util```: utils.
  - ```train.py```: train model.
  - ```deploy.py```: test model.

## Train

You should modify the options.py firstly, and then:

```python
python train.py
```

## Test

You should modify the options.py firstly, and then:

```python
python deploy.py
```

## Reference

- [pytorch-multi-label-classifier](https://github.com/pangwong/pytorch-multi-label-classifier)

  
