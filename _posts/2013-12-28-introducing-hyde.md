Getting Started With Numpy {#getting-started-with-numpy .p-name}
==========================

::: {.section .p-summary field="subtitle"}
What is numpy?
:::

::: {.section .e-content field="body"}
::: {.section .section .section--body .section--first .section--last name="8e8b"}
::: {.section-divider}

------------------------------------------------------------------------
:::

::: {.section-content}
::: {.section-inner .sectionLayout--insetColumn}
### Getting Started With Numpy {#c7a5 .graf .graf--h3 .graf--leading .graf--title name="c7a5"}
:::

::: {.section-inner .sectionLayout--fullWidth}
![Photo by [Andrew
Schultz](https://unsplash.com/@beatthegrind?utm_source=medium&utm_medium=referral){.markup--anchor
.markup--figure-anchor}
on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral){.markup--anchor
.markup--figure-anchor}](https://cdn-images-1.medium.com/max/2560/0*TZ3NyQsdkS1MCjOD){.graf-image}
:::

::: {.section-inner .sectionLayout--insetColumn}
#### What is numpy? {#7b38 .graf .graf--h4 .graf-after--figure name="7b38"}

Numpy (remember it as numerical python) is a package which is used for
scientific computing in python. It contains data structures which could
handle many operations which are typically encountered in scientific
computation in a fast and efficient way. To name a few, it can perform
the following operations

-   [Matrix operations.]{#38ef}
-   [Sorting & shape manipulations.]{#809a}
-   [Simulating randomness (well psedo-randomness).]{#5d2c}
-   [Basic linear algebra (dot product, cross product, inverse
    etc.).]{#61bd}
-   [Basic statistical operations (mean, standard deviation, median
    etc.).]{#3ef2}

#### Why numpy? {#934a .graf .graf--h4 .graf-after--li name="934a"}

A simple answer would be

> Numpy is fast, efficient, has a wide range of utility functions.

However, a more expository version is as follows.\
Numpy arrays are vectorized. Vectorized implementation meaning the user
doesn't have to write any looping constructs like for or while for
repetitive operations. All of it is handled behind the scenes in an
optimized, pre-compiled C code.

Vectorized implementations are computationally cheap than loops,
therefore they are faster. x below is a numpy array of 5 numbers.

![Comparing Looped vs vectorized implementations for multiplying array
with
a scalar](https://cdn-images-1.medium.com/max/800/1*vdCJaP7NkSXFGGKwBTrBbw.png){.graf-image}

In the absence of loops, written code more closely resembles the natural
language. This makes the code easier to comprehend. For eg. compare
matrix multiplication operation in vectorized v/s non-vectorized
implementation below and decide for yourself. x and y are matrices of
dimensions 1000 \* 10 and 10 \* 1000 respectively.

![Vectorized implementations --- Pythonic and
Intuitive](https://cdn-images-1.medium.com/max/800/1*nhN3r2jsoPFJgS08MOv96A.png){.graf-image}

#### Installing numpy {#d3f1 .graf .graf--h4 .graf-after--figure name="d3f1"}

You should have python and pip installed for installing numpy to your
machine. pip is a package manager which is used for installing packages
in python. It stands for either *Pip Installs Packages* or *preferred
installer program*.

You can visit [this
page](https://www.python.org/downloads/){.markup--anchor
.markup--p-anchor} on the official python website for installing python
for any OS --- Windows, Linux (Ubuntu, LinuxMint etc.), MacOS and
others.

This [YouTube
Video](https://www.youtube.com/watch?v=Ko9b_vC6XY0){.markup--anchor
.markup--p-anchor} is a good resource to install pip on your system.

Once you have them installed, you can open a command prompt in windows
or a terminal in mac/ubuntu and type

``` {#ac82 .graf .graf--pre .graf-after--p name="ac82"}
pip install numpy
```

This will automatically install the latest version of numpy to your
local system. To verify if numpy has installed properly, in your
shell/command prompt, type the following commands. If they execute
successfully, that means you're good to go!

``` {#88c9 .graf .graf--pre .graf-after--p name="88c9"}
python
import numpy
print(numpy.__version__)
```

-   [The first command \`python\` opens the python interpretor in the
    terminal/command prompt.]{#57bf}
-   [The second command imports the package so that we can have it for
    our use.]{#04d9}
-   [The third command prints the version of the numpy package we
    installed.]{#ad86}

*In the next post, we will get started with numpy arrays.You can find
the next post below*

[**Basics of Numpy
Arrays**](https://medium.com/@nayakvinayak95/numpy-arrays-35b171a18cae){.markup--anchor
.markup--p-anchor}

#### References {#1ec8 .graf .graf--h4 .graf-after--p name="1ec8"}

-   [[Numpy official
    website](https://numpy.org/doc/stable/user/setting-up.html){.markup--anchor
    .markup--li-anchor}]{#53fc}
:::
:::
:::
:::

By [Vinayak Nayak](https://medium.com/@nayakvinayak95){.p-author
.h-card} on [August 17, 2020](https://medium.com/p/22e8ce42b9ea).

[Canonical
link](https://medium.com/@nayakvinayak95/getting-started-with-numpy-22e8ce42b9ea){.p-canonical}

Exported from [Medium](https://medium.com) on February 24, 2021.