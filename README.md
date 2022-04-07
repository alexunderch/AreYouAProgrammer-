# AreYouAProgrammer-
Here we are! Imagine that you're a neural network engineer who wants to become a software engineer (to have some tremendous salary, I guess).
So, you have just been employed to help some high-tech company to design and trade AI (hype!) products. Beforehand, here worked 3 programmers but they were fired (hmmm?).
Maybe, for simplicity, the startup wants to create his own MNIST-token and builds their own system to recognize hand-written digits. Why not to take somehing already working? Just because.

The very first programmer designed `InterfaceA` -- a simple `Pytorch` learner without any doctsrings and flexible parameters. Of course, no Convolutional networks because that's faster on CPU. But he worked so hard that burned out and was released.

The next one was his mad brother. He wrote the prerocessing pipeline but in some reason did this in `tensorflow` (typical `Keras` enjoyer, we see), refused to document and support his own code, and consequenly was fired.

The third specialist was super smart: he designed the whole pipeline (his metodology is pretty adored by the Chefs) but understood that different parts are incompatible, so just changed his workplace to Booble (don't mess it with Google, please)

So, here you are!
You task is:
* `InterfaceC.py` is the file with the pipeline (you can actually rename files), dont change its structure but fill the methods with the necessary code -- you should demonstrate training, validation and testing performance to your boss.
* `interfaceB.py` (change it to `torchvision` with the nitty-gritty transforms, your boss mignt want to see some weird and cringe ones)
* `interfaceA.py` -- your boss wants to look at the difference between FC and convolutional architectures, so make to learning classes (all the loop should incorporated in one class) and inherit it from a base class (maybe it will be smth different learning tasks afterwards)

Please, add more comments and docstrings like
```Python
def function(param1: type, param2: type,...) -> type:
  """Params:
    :param -- type, description
    Returns:
    :...
  """
  ...
```
P.s. if you are supersmart, inherit all your classes from [dataclass](https://docs.python.org/3/library/dataclasses.html) and make some tests.
Looking forward to your pull requests,
with love,
Booba
