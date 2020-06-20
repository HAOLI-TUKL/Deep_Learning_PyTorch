# Deep_Learning_PyTorch
deep learning implementation including Fizzbuzz game, image classification, transfer learning , style transfer and GAN based on PyTorch using Python

## FizzBuzz Game
FizzBuzz game is a simple game. We count from 1 to 99 and say fizz, buzz and fizzbuzz respectively when the multiple of 3 or 5 or 15 occurs.
A simple two layer network is traine based on PyTorch for this task, and the result is below.
```
['1', '2', 'fizz', '4', 'buzz', 'fizz', '7', '8', 'fizz', 'buzz', '11', 'fizz', '13', '14', 'fizzbuzz', '16', '17', 'fizz', '19', 'buzz', 'fizz', '22', '23', 'fizz', 'buzz', '26', 'fizz', '28', '29', 'fizzbuzz', '31', '32', 'fizz', '34', 'buzz', 'fizz', '37', '38', 'fizz', 'buzz', '41', 'fizz', '43', '44', 'fizzbuzz', '46', '47', 'fizz', '49', 'buzz', 'fizz', '52', '53', 'fizz', 'buzz', '56', 'fizz', '58', '59', 'fizzbuzz', '61', '62', 'fizz', '64', 'buzz', 'fizz', '67', 'fizz', 'fizz', 'buzz', '71', 'fizz', '73', '74', 'fizzbuzz', '76', '77', 'fizz', '79', 'buzz', 'fizz', '82', '83', 'buzz', 'buzz', '86', 'fizz', '88', '89', 'fizzbuzz', '91', '92', 'fizz', '94', 'buzz', 'fizz', '97', '98', 'fizz']
```
## Image Classification based on CNN and transfer learning
Mnist datasets are used for this task and a CNN network is trained. Mnist dataset is a dataset containing lots of handwriting numbers. For transfer learning, when sometimes we want to reuse the weights of a pretrained network, only the linear layer at the end of the network is trained and other weights are kept unchanged in this case, because these unchanged layers act as feature extractor, which are helpful for new task. In comparsion with training from begining, transfer learning can quickly achieve higher accuracy.
<img src="https://github.com/HAOLI-TUKL/Deep_Learning_PyTorch/blob/master/images/transfer1.png" alt="none" width="500" height="313" align="bottom" />

## Style Transfer
Style transfer combines the contain of an image and the style of another image together. For this task, 0, 5, 10, 19,28 layers of vgg19 are used for feature extractions. Keep in mind that no network is trained for this task but a target/combined image is trained. The key point for this task is how to define feature loss and content loss.
the feature loss is defined as  
<img src="https://github.com/HAOLI-TUKL/Deep_Learning_PyTorch/blob/master/images/style1.png" alt="none" width="500" height="313" align="bottom" />. 
the content loss is defined as  
<img src="https://github.com/HAOLI-TUKL/Deep_Learning_PyTorch/blob/master/images/style1.png" alt="none" width="500" height="313" align="bottom" />. 


