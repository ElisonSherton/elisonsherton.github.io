---
layout: post
title: Pagination in Streamlit
published: true
categories: ['Visualization']
---

This post is aimed toward demonstrating the use of session state in `Streamlit` for storing info about certain variables and prevent them from updating across runs. But what is `Streamlit` ?

`Streamlit` is a python library which is aimed to make the process of building web applications to showcase your work very easy for python developers. Until the arrival of this package `Flask` and `Django` were the goto libraries which developers chose to use in order to develop and deploy their application over the web; however both these frameworks required the user to write HTML/CSS code to render their work as a web-app. `Streamlit` abstracts all this and provides an easy pythonic interface for adding custom components like sliders, dropdowns, forms, textboxes etc. while also allowing people to add their own custom HTML/CSS components. If you’re new to Streamlit, their [official documentation](https://docs.streamlit.io/en/stable/getting_started.html) is very good to get you started with the same.
In this post, I am going to be discussing how we could build a component which could help implement pagination support as it is not something that is provided out of the box and needs a small hack of `SessionStates`.

<hr>

## What is Pagination?

Wikipedia defines pagination as process of dividing a document into discrete pages, either electronic pages or printed pages.

So basically if you have a lot of content which you want to display on your app/web-page, you can break them into a number of component pages which could be shown one by one and the trigger to switch between pages could be a button like a `Next` button or a `Previous` button.

On many web pages you can see pagination pay a key role for eg.

- On any e-commerce site where there’s a lot of products to display, they’re typically showed in a page-wise fashion.

- The search results for Google Search of a given query are always visible in pages and not all at once.

Streamlit’s current version doesn’t provide out of the box support to tackle this use-case but in this post we’re going to talk about a hack which will enable us to perform pagination in a neat manner.

<hr>

## The Problem

If we look at a simple demo `Streamlit` code for displaying a dataframe to our screen, we can see the code is as follows:

{% gist ce437c406c5b4c3ab743e814d8edbd85 %}

The issue that we face here is that if we want to do pagination, we will have to define a page number and if we initialize it in here.

Next we need to create a button called `Next` and `Previous` which will move across the next or previous entries in the dataframe. But when we link to the variable page_number to these buttons, every time page number is clicked or there’s any component’s state change in the session, the variable gets re-initialised to it’s starting value and we never end up dynamically moving across pages. So, suppose we had a code like this

{% gist 4e223bbc218a714b030074b9cc481c63 %}

We have created a button next and a button previous which when clicked change the page number based on some conditional checks (i.e. don’t overshoot in case of first and last pages; roll back to the last/first pages respectively as case be). If we use this code as is and display the dataframe in the app, our output would be as follows:

![](https://miro.medium.com/max/700/1*dycXI8P3_M_3IPp5OSFRUw.gif)

When we click on next, the page number gets reinitialised to 0. That’s why we are never able to move ahead of page 1. The page number reinitialises to 0, gets incremented by 1 and that’s it, that happens in a loop.

Similarly when we click on previous, page number gets reinitialised to 0 and then decrements to the last page and this keeps happening perpetually!

<hr>

## The Solution

There’s [this piece of code](https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92) which is generously lent to the users of streamlit by [Thiago Teixeira](https://github.com/tvst) who is the co-founder of streamlit that allows us to create objects which we can track in a session by not overwriting/reinitializing them over and again.

You basically copy the code given in the piece of code above to a file called SessionState.py and keep it in the same work directory as the application file. Next you just have to wrap the variable who you want to stop from getting re-initialised inside a Session State object by using the get method of the same as shown below.

{% gist 0136e8054777e159bd7d387a870e91ed %}

In case you have more variables to keep track of you could write

```python
import SessionState
ss = SessionState.get(page_number = 0, var_1 = True, var_2 = "Streamlit demo!")
```
And it would also work fine. But a caveat here is that

> You can only have one session state object per application

This is important to note because otherwise if you’re gonna try to initialize multiple states with multiple variables/attributes, that will end up not working and throw an error. So you have to be cautious about which variables to track dynamically and which ones to track via sessions and encircle the latter inside of the one and only SessionState object that you’re gonna create. The result of the above code works perfectly and looks as follows

![](https://miro.medium.com/max/700/1*yD3mAwdfY_ADk7r1kFKujw.gif)

<hr>

## References

1. [Github link for source code of this app](https://github.com/ElisonSherton/streamlit-next-button-demo)
2. [Dataset reference — auto-mpg.csv from Kaggle](https://www.kaggle.com/uciml/autompg-dataset)