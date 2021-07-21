---
layout: post
title: W&B Fastbook Sessions Week 6 Summary
published: true
categories: ['fastbook', 'Deep Learning']
---


# Introduction

In this post, I would like to summarize the concepts covered in week 6 of the fastbook sessions. We started reading the `05_pet_breeds` chapter and completed the first half of the same. It was a an awesome session conducted by [Aman Arora](https://twitter.com/amaarora) from [Weights & Biases](https://twitter.com/weights_biases). 

The session is available for streaming on YouTube and below's the link for the same.

<iframe width="560" height="315" src="https://www.youtube.com/embed/bvtr_1TN6MI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

> In this session we started digging into depth over Image Classification problems. We looked at the full version of PETs datasets considering the breeds into account this time. In the first section we discussed about some fastai convenience structures like `untar_data`, `L` etc., looked at `Regexes` for labelling our data and `Cross Entropy Loss` which is one of the most common loss functions used for classification across lots of tasks. 

I would like to cover the following concepts in the rest of the post.

1. Regular Expressions
2. Some convenience functions in fastai

<br>

# Regular Expressions (RegEx)

Regexes short for regular expressions are a language in themselves. They are a language which specifies a set of rules for representation of common language strings like English in a particular syntax.

This can help us find if a given word/phrase/sentence(more generally strings) contains a particular pattern that we want to be there and more powerfully, extract such patterns whenever they're encountered from such strings.

Let us play around with the pets dataset URLs using regex to understand the importance of this language and subsequently how this could help us in Deep Learning.

Here is a really good one stop resource for understanding different characters/patterns in regex which can help create our custom query patterns for string matching.

[Data Quest Regex Cheatsheet](https://www.dataquest.io/wp-content/uploads/2019/03/python-regular-expressions-cheat-sheet.pdf)

In python, we have a module called `re` which comes in pre-built that could be used to leverage regular expressions.

Consider the PETs dataset for example. Each of the file names have been encoded meticulously to provide certain information. Let's have a cursory glance at the files.

```python
from fastai import *
from fastai.vision import *

path = untar_data(URLs.PETS)
(path/"images").ls()
```

![PETs Image links](https://i.imgur.com/nSxSZ4r.png)

After looking at the files and the documentation of the [dataset here](https://www.robots.ox.ac.uk/~vgg/data/pets/) we come to know the following

- The name of all dog breed files start with a small letter
- The name of all files is in the format <breed>_<number>.jpg
- If the breed is a multiple word breed, then the breed will be represented by _ in between words

This is a perfect usecase for regexes to perform certain tasks. Let us write some expressions for performing certain tasks!

## Using regex to get all the cat files

As we saw above all cat filenames start with an uppercase first letter in the filename. So, we will utilize that and write a regex to filter those files as follows

> cat_pattern = r"[A-Z].+"

- [A-Z] : this means that the pattern to match must have a capital letter as it's first alphabet. 
- .+ : this means that match and pick up any character that follows the capital letter.

![Imgur](https://i.imgur.com/6yvp9Vm.png)

As we can see, there are a total of 2403 cat files in our directory

Since we know here that all our cat files start with caps and dogs with small, this simple regex works well. However if we had some more caps letters, it would pick up multiple patterns with this regex, then in that case, we need to modify the regex as follows

> modified_cat_pattern = r"^[A-Z].+"

Now this means that the string to be matched MUST START WITH a capital letter i.e. `string_var[0]` must be an uppercase character. This is a more specific pattern and will also yield similar results as previous usecase for this problem...

## Using regex to get all the cat breeds

Mind that now we need the cat-breeds. The regex defined above is capable of capturing all cat-filenames so we're partially there. However we don't need the digits and the extension at the end of the filenames for getting breeds... 

We can add what is called a `capture_group` to only retrieve parts of the regex that we've defined.

> catbreed_pattern = r"(^[A-Z].+)_\d+"

Notice that now we used the modified_cat_pattern from above and we added parantheses around the pattern and added `_\d+`. This means that match a pattern which ends in `_xxxx..` where x is a number and can be any number of numbers more than 1 (+ means match 1 or more times)

![Imgur](https://i.imgur.com/W2kn8hC.png)

Then after extracting only the unique breed names out of the list, we observe that there's a total of 12 cat breeds in our dataset. 

*This is also a part of EDA i.e. exploring our data. So what this means is that regexes can come in very handy when trying to explore and get a sense for what all is there in our data*.

## Using regex to get all the dog breeds

Dog filenames start with a lowercase letter and have any number of characters followed by an _ and some digits. This pattern could be encoded as follows and we can define a capture group around the dog phrases.

> dogbreed_pattern = r"(^[a-z].+)_\d+"

This means fetch me all the matches in a filename which start with a lowercase alphabet followed by some characters and they should end with `_xxxx...` where x is a digit and can extend to any number of numbers.

![Imgur](https://i.imgur.com/Kkg5jHE.png)

fastai defines a very convenient function called `uniqueify` which we'll discuss below. I have used that in both the above usecase and this usecase to find out a unique list of all the dogbreeds from amongst the list of all pattern-matches obtained from individual filenames. 

In this case we see that there's a total of 25 dog-breeds. 

This means we have a combined total of 37 breeds across both cats and dogs.

## Getting all cats and dogs valid files

We only want to be dealing with file-formats that could be handled by Pillow and we know our dataset is expected to be all `.jpg` image files. So we can write a regex to confirm this.

> dog_files = r"^[a-z].+_\d+.jpg$"
> cats_pattern = r"^[A-Z].+_\d+.jpg$"

The first part of the pattern remains as is but we add a condition at the end which is the file name must end with a `.jpg` extension as symbolized by the dollar sign at the end of the regex pattern.

We can also combine the above two patterns into one as follows

> cats_or_dogs_pattern = r"^[a-zA-z])+_\d+.jpg$"

![Imgur](https://i.imgur.com/hbCT6nQ.png)

This tells us that there's three files which don't have `.jpg` but a `.mat` extension to them. This way regex can also be used to clean our data if there's some inconsistencies in the same.

Although looking at these small examples might not help us appreciate the importance of regex, it's pretty much a life saver when it comes to tasks involving preprocessing text data. In Natural Language Processing, figuring out patterns based on regex is very common and at that time we understand the necessity of regex.

In the session we also looked at a set of some functions which are super-convenient and pre-defined by fastai for us to use out of the box. Let's have a look at some of them next.

<br>

# Convenience functions in fastai

While going through the chapter, I observed that Jeremy and the fastai team have written pretty neat functions which come in use a lot when dealing with any operations related to data preprocessing in our day to day tasks involving DL/ML/DS etc. I will go through some functions which I found really helpful.

These functions are defined in the `fastcore.basics` package which gets imported when we import everything from vision package.

## L -> Equivalent of List in fastai

fastai has introduced a really nice wrapper around python's list class called L. It is functionally the same as a list but it's more convenient to work with. Let me show you with an example.

![Imgur](https://i.imgur.com/uOeyGTu.png)

If we look at the default list in a jupyter shell, it just extends in the output block to show all items. Many times we just want to look at a few items in our list, `L` triumphs over `list` in such scenarios; by default it only shows the top 10 items in the list when we display it in the output shell.

Also, we are most of the times also interested in knowing how many elements are there in the list. `L` is printed out in such a way that the number of elements is displayed first with a `#` and then the first 10 elements are printed. This is pretty cool! Because everytime I don't have to print out `len(list)` to look at how many elements are there in my list. It's just there in front of me!

All the operations that could be performed on lists like slicing, indexing etc. can also be performed with L. So, we can call L as python lists supercharged with wings! :p

## listify/tuplify

These are functions which could be used to turn any collections into lists and tuples as need be.

![Imgur](https://i.imgur.com/l7hNzDs.png)

Many a time during our data processing, we tend to work with different collections like sets to do all sorts of things like set algebra (intersection, union, difference, complement etc.) Eventually we want to bring it back into a list/tuple to work with. Then these functions really come in handy.

Also sometimes, we want to retrieve all the keys of a dictionary in a list. We can use either of these functions to do that and it's slightly better than `dict.keys()` because it directly returns a list whereas the latter returns a key map which has to be converted to a list.

## uniquefy/last_index

The best part about the naming conventions in fastai is they're very intuitive and eponymous. 

![Imgur](https://i.imgur.com/UgnkxTB.png)

`uniqueify()` provides a list of all the unique elements in a collection. When doing some EDA if you wish to see what are the unique levels in a categorical variable or if you want to figure out how many unique integer states are there in an integer list which could be representative of some real world state etc. this function is very handy.

Also sometimes, we would like to know when is the latest occurrence of a particular item; especially in time-series analysis we would like to figure out the most recent occurence of an event etc. At that time `last_index()` function comes in very handy!

## Dictionary filtering functions

These set of functions are my favorite of the lot.

![Imgur](https://i.imgur.com/AQNrE0W.png)

Although lists and tuples and L are good data structures to store sequences, many a time we can't be content with numbers as indices to look for things. Hence the dictionary as a python data structure, in my honest opinion, at least for the work that I do, is probably the most widely used data structure.

Having these three functions very handy makes my code clean, intuitive and very convenient.

Now, let's say I have a dictionary of products as keys and their costs as values as shown above. Each product is also structured in a way `<product_type>: <product_name>`. This is a very possible real world scenario.

We can use `regex` + `filter` to do many things and I'll demonstrate the same below.

### Filtering at a key level

Let's say we only want to filter those products from the dictionary which are fruits

```python
# Create a simple dictionary
d = {"Fruit: Apple":5, "Fruit: Bananas":4, "Menu: Apple Pie":34, "Ingredient: Butter": 32, "Ingredient: Salt": 5}

# Filter out only fruits from the dictionary
fruit = lambda query: re.match(r"Fruit: \w+", query)
filter_keys(d, fruit)
```

Python has a powerful one-line function concept called `Lambda Functions` which are super useful. We are defining a function to match the regex for our *Fruit:* product and then if we pass that as an argument to the `filter_keys` function along with the dictionary, the function does the job of filtering those elements whose keys satisfy some condition as defined in the function.

### Filtering at a value level

Similar to the above usecase, now let's say we want to query something at a value level i.e. products which cost less than 10 units.

```python
# Create a simple dictionary
d = {"Fruit: Apple":5, "Fruit: Bananas":4, "Menu: Apple Pie":34, "Ingredient: Butter": 32, "Ingredient: Salt": 5}

# Filter out only items with less than 10 units cost
cost = lambda query: query < 10
filter_values(d, cost)
```

We can define a similar lambda function assuming that we'll get cost as the input as shown above. Then on passing the function and the dictionary to the `filter_values` method, we can get those elements whose cost is less than 10 units.

### Filtering at a key-value level

Let us say we want to filter out items based on a condition for the key and a condition for the value as well for eg. filter out all ingredients which don't exceed 10 units.

```python
# Create a simple dictionary
d = {"Fruit: Apple":5, "Fruit: Bananas":4, "Menu: Apple Pie":34, "Ingredient: Butter": 32, "Ingredient: Salt": 5}

# Filter out ingredients with cost less than 10 units
cheap_ingredient = lambda k, v: re.match(r"Ingredient: \w+", k) and v < 10
filter_dict(d, cheap_ingredient) 
```

`filter_dict` is built exactly for this. Similar to the above two functions it takes in a dict and a function but the function needs to be written carefully here.

> The function must first expect the keys and then the values not the other way round. 

Then based on these key and value we can define a function of any level of complexity (it need not be a lambda function as I have written above; it's only to demonstrate for this simple application). 

I hope you had fun reading through the post and hope you learned something new today! :)  I would be glad to [connect with you on Twitter](https://twitter.com/ElisonSherton). If you have any comments/suggestions/thoughts, feel free to comment below or reach out to me on Twitter. 

If you liked what you read, feel free to check out [my other posts here.](https://elisonsherton.github.io/categories.html).

# References

1. [fastbook chapter 5](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb)
2. [wandb fastbook sessions link](https://www.youtube.com/embed/bvtr_1TN6MI)
3. [Github code for the application created in the post](https://github.com/ElisonSherton/fastbook_sessions/tree/master/ch5PetBreeds)
4. [wandb forums for week 4](https://wandb.ai/aarora/discussions/FastBook-Reading-Group-Week-6--Vmlldzo4NTMwMDM?galleryTag=forum)
5. [Regular Expressions Python Cheatsheet](https://www.dataquest.io/wp-content/uploads/2019/03/python-regular-expressions-cheat-sheet.pdf)