---
layout: post
title: Getting Started With Numpy
---

<div class="section p-summary" field="subtitle">

What is numpy?

</div>

<div class="section e-content" field="body">

<div class="section section section--body section--first section--last"
name="8e8b">

<div class="section-divider">

------------------------------------------------------------------------

</div>

<div class="section-content">

<div class="section-inner sectionLayout--insetColumn">

### Getting Started With Numpy

</div>

<div class="section-inner sectionLayout--fullWidth">

<figure>
<img src="https://cdn-images-1.medium.com/max/2560/0*TZ3NyQsdkS1MCjOD" class="graf-image" alt="Photo by Andrew Schultz on Unsplash" /><figcaption aria-hidden="true">Photo by <a href="https://unsplash.com/@beatthegrind?utm_source=medium&amp;utm_medium=referral" class="markup--anchor markup--figure-anchor">Andrew Schultz</a> on <a href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral" class="markup--anchor markup--figure-anchor">Unsplash</a></figcaption>
</figure>

</div>

<div class="section-inner sectionLayout--insetColumn">

#### What is numpy?

Numpy (remember it as numerical python) is a package which is used for
scientific computing in python. It contains data structures which could
handle many operations which are typically encountered in scientific
computation in a fast and efficient way. To name a few, it can perform
the following operations

-   <span id="38ef">Matrix operations.</span>
-   <span id="809a">Sorting & shape manipulations.</span>
-   <span id="5d2c">Simulating randomness (well
    psedo-randomness).</span>
-   <span id="61bd">Basic linear algebra (dot product, cross product,
    inverse etc.).</span>
-   <span id="3ef2">Basic statistical operations (mean, standard
    deviation, median etc.).</span>

#### Why numpy?

A simple answer would be

> Numpy is fast, efficient, has a wide range of utility functions.

However, a more expository version is as follows.  
Numpy arrays are vectorized. Vectorized implementation meaning the user
doesn’t have to write any looping constructs like for or while for
repetitive operations. All of it is handled behind the scenes in an
optimized, pre-compiled C code.

Vectorized implementations are computationally cheap than loops,
therefore they are faster. x below is a numpy array of 5 numbers.

<figure>
<img src="https://cdn-images-1.medium.com/max/800/1*vdCJaP7NkSXFGGKwBTrBbw.png" class="graf-image" alt="Comparing Looped vs vectorized implementations for multiplying array with a scalar" /><figcaption aria-hidden="true">Comparing Looped vs vectorized implementations for multiplying array with a scalar</figcaption>
</figure>

In the absence of loops, written code more closely resembles the natural
language. This makes the code easier to comprehend. For eg. compare
matrix multiplication operation in vectorized v/s non-vectorized
implementation below and decide for yourself. x and y are matrices of
dimensions 1000 \* 10 and 10 \* 1000 respectively.

<figure>
<img src="https://cdn-images-1.medium.com/max/800/1*nhN3r2jsoPFJgS08MOv96A.png" class="graf-image" alt="Vectorized implementations — Pythonic and Intuitive" /><figcaption aria-hidden="true">Vectorized implementations — Pythonic and Intuitive</figcaption>
</figure>

#### Installing numpy

You should have python and pip installed for installing numpy to your
machine. pip is a package manager which is used for installing packages
in python. It stands for either *Pip Installs Packages* or *preferred
installer program*.

You can visit
<a href="https://www.python.org/downloads/" class="markup--anchor markup--p-anchor">this page</a>
on the official python website for installing python for any
OS — Windows, Linux (Ubuntu, LinuxMint etc.), MacOS and others.

This
<a href="https://www.youtube.com/watch?v=Ko9b_vC6XY0" class="markup--anchor markup--p-anchor">YouTube Video</a>
is a good resource to install pip on your system.

Once you have them installed, you can open a command prompt in windows
or a terminal in mac/ubuntu and type

``` graf
pip install numpy
```

This will automatically install the latest version of numpy to your
local system. To verify if numpy has installed properly, in your
shell/command prompt, type the following commands. If they execute
successfully, that means you’re good to go!

``` graf
python
import numpy
print(numpy.__version__)
```

-   <span id="57bf">The first command \`python\` opens the python
    interpretor in the terminal/command prompt.</span>
-   <span id="04d9">The second command imports the package so that we
    can have it for our use.</span>
-   <span id="ad86">The third command prints the version of the numpy
    package we installed.</span>

*In the next post, we will get started with numpy arrays.You can find
the next post below*

<a href="https://medium.com/@nayakvinayak95/numpy-arrays-35b171a18cae" class="markup--anchor markup--p-anchor"><strong>Basics of Numpy Arrays</strong></a>

#### References

-   <span
    id="53fc"><a href="https://numpy.org/doc/stable/user/setting-up.html" class="markup--anchor markup--li-anchor">Numpy official website</a></span>

