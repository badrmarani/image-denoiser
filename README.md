# Variational Image Denoising

In this project, we implemented Total Variation Minimization, in image denoising, using Fixed-Point algorithm and Projected Gradient Descent algorithm.

## Installation
### Dependencies

All of these Python dependencies, can be installed with `pip install -r requirements.txt` inside the root Image folder.

## Quick example usage

Evaluate Chambolle1 (Semi-implicit gradient descent algorithm applied to ROF) on `girl.jpg`.

``` shell
python main.py --max-iter 50 --sigma 50 --L 60 --epsilon 1e-4 --step-size .2
```

**Note:** By default, images are denoised using `chambolle1` method.

### Output
```shell
Processing : houses.jpg
> Variational Method: cham1
   Number of iterations    Mean    Noise    Lambda    Step size    Epsilon
-----------------------  ------  -------  --------  -----------  ---------
                     50       0       50        60          0.2       1e-4
First layer
100%|##########################################################| 50/50 [00:36<01:30,  1.61it/s]
Second layer
100%|##########################################################| 50/50 [00:40<02:07,  1.14it/s]
Third layer
100%|##########################################################| 50/50 [00:41<01:41,  1.40it/s]

Time spent is: 120.41364026069641

Processing : lenna.jpg
100%|##########################################################| 50/50 [00:08<00:26,  5.66it/s]

Time spent is: 128.97879600524902
```
