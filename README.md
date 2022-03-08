# Variational Image Denoising

## Installation
### Dependencies

All of these Python dependencies, can be installed with `pip install -r requirements.txt` inside the root Image folder.

### Install from source
1. Navigate to your desired installation directory and download the GitHub repository:
``` shell
git clone https://github.com/maranibadr/image
```

1. Navigate to the top-level folder (should be named Image and contain the file `setup.py`) and run `setup.py`:
``` shell
cd Image
python setup.py install --user
```

That's it!

## Quick example usage
Evaluate Chambolle1 (Semi-implicit gradient descent algorithm applied to ROF) on a `face` image.

``` shell
python main.py --max-iter 3000 --sigma 50 --L 60 --epsilon 1e-4 --step-size .2
```

**Note:** By default, images are denoised using `chambolle1` method.


### Output
```
Processing : houses.jpg
First layer
100%|##################################################################################| 50/50 [00:36<01:30,  1.61it/s]
Second layer
100%|##################################################################################| 50/50 [00:40<02:07,  1.14it/s]
Third layer
100%|##################################################################################| 50/50 [00:41<01:41,  1.40it/s]

Time spent is: 120.41364026069641

Processing : lenna.jpg
100%|##################################################################################| 50/50 [00:08<00:26,  5.66it/s]

Time spent is: 128.97879600524902
```
#### Results!!!
![alt text](./data/test.jpg)
