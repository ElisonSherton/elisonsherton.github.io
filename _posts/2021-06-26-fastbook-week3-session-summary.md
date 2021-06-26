---
layout: post
title: W&B Fastbook Sessions Week3 Summary
published: true
categories: ['fastbook', 'Deep Learning']
---

# Introduction

This post is aimed to summarise what I learned from the `fastbook reading session` with [Aman Arora](https://twitter.com/amaarora) from [Weights & Biases](https://twitter.com/weights_biases). It is a study group which has come into effect thanks to the collective effort of Aman and the organization Weights and Biases for which I am really grateful. The entire session is recorded and can be viewed below

<iframe width="560" height="315" src="https://www.youtube.com/embed/rmOqCO7c8pw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This post summarises the contents of week 3 with the help of a personal project that I did to replicate the concepts taught in this session. I hope you find this helpful :) 

# Creating your very own dataset

Deep Learning/ Machine Learning, unlike traditional rules-based programming is a method which learns the rules from data which is fed to it. For the classification problem we need to provide labelled data i.e. tags/categories associated with the datapoint.

Although there are many toy datasets like `MNIST`, `FashionMNIST`, `Oxford-IIIT Pets Dataset` etc., in the real world scenario you'll almost always have to curate your own dataset and there's several ways to go about it. Most of them involve scraping data from the internet in order to build a labelled dataset; however the scraping process might be very intimidating for a beginner and so the good folks at `fastai` have created very handy functions to obtain datasets from search engines like `Bing` and `DuckDuckGo`. 

As a toy project, I am building a cartoon classifier. As a kid I grew up watching the `Road Runner` show and `Shinchan` quite a lot, so out of that nostalgia, I wanna create a classifier which helps tell 4 different characters namely `Bugs Bunny`, `Donald duck`, `Mickey Mouse` and `Shinchan` apart.

So, let's get down to the code then. We need to import the fastai library and the vision package libraries from the same.

```python
from fastai import *
from fastai.vision.all import *
```

Next, we shall use the function provided by fastai to download images from duck duck go search engine given a search term. The function is as follows

```python
def search_images_ddg(term, max_images):
    "Search for `term` with DuckDuckGo and return a unique urls of about `max_images` images"
    
    assert max_images<1000
    
    url = 'https://duckduckgo.com/'
    res = urlread(url,data={'q':term})
    searchObj = re.search(r'vqd=([\d-]+)\&', res)
    assert searchObj
    
    requestUrl = url + 'i.js'
    params = dict(l='us-en', o='json', q=term, vqd=searchObj.group(1), f=',,,', p='1', v7exp='a')
    urls,data = set(),{'next':1}
    
    while len(urls)<max_images and 'next' in data:
        try:
            data = urljson(requestUrl,data=params)
            urls.update(L(data['results']).itemgot('image'))
            requestUrl = url + data['next']
        except (URLError,HTTPError): pass
        time.sleep(0.2)
    
    return L(urls)
```

> Please note that if you're using the `search_images_ddg` function from fastbook directly, it might throw an error. This code above has been modified to remove the `.decode()` from the `urlread()` function as the function already decodes it in the latest version of the package. You can refer [this fastai forums thread](https://forums.fast.ai/t/attributeerror-str-object-has-no-attribute-decode/87164/2) for further clarification.

Now, what I do is I define a list of the characters for which I want the data to be scraped and store their corresponding urls retrieved using the function above as a dictionary that maps a character to a list of it's relevant links.

```python
# Define a list of my favorite cartoons
cartoons = L(["bugs bunny", "donald duck", "mickey mouse", "shinchan"])

# Define a dictionary which maps all the cartoons to their respective links as returned 
# from the function above and collect data from the same
cartoonsLinks = {}
for cartoon in progress_bar(cartoons, comment = "Extracting image links"):
    cartoonsLinks[cartoon] = search_images_ddg(cartoon, max_images = 200)
```

Once we have this list, we shall download the images to a folder on our disk using the code below.

```python
# Now that we have the links for every item, let's download the images
path = Path("/home/vinayak/cartoons")

# Do this only if the cartoons folder doesn't exist
if not path.exists():
    path.mkdir()
    
    # Loop over each cartoon links
    for cartoon in progress_bar(cartoonsLinks.keys(), comment = "Downloading images..."):
        # Create a folder to hold the cartoons images (if a folder already exists, don't raise an error) 
        dest = (path/cartoon)
        dest.mkdir(exist_ok=True)
        
        # Retrieve the cartoons links and download the images to the specified path
        results = cartoonsLinks[cartoon]
        download_images(dest, urls=results, timeout=2)
```

Once we do these steps, we will see that not all the image links which were reeturned by duck duck go search engine were downloadable. If we print out the stats exactly we see this.

![](https://i.imgur.com/lVsDLbO.png)

It is possible some image links were broken or didn't allow the download or had a timeout before getting downloaded anything could've gone wrong. So it's better to scrape links for 10% or 20% more datasize than what you expect your final datasize should be and then you can prune it down always if it is too much data...

Even after we download the images, it doesn't guarantee that all images which are downloaded are usable. We can then use the `verify_images` function from vision package in fastai to delete such images. These images could be truncated images or RGBA images or webp images etc. basically unloadable images. In my downloaded dataset, there were around 30 unusable images which were deleted using the `unlink` function from the Path module.

![](https://i.imgur.com/N5eTiIx.png)   

So we now have a fully workable dataset which we can use to train a SOTA Deep Learning model for image classification.

# Building a classifier on your own dataset

What we see as images on our screens are interpreted by the computer as matrices of decimal numbers. So, in order to load the images into a way that the computer can understand fastai provides couple of classes: the `DataBlock` and the `DataLoader`. It is essential to understand these two classes in detail to be able to feed our data to the model and let it do it's weight updates and come up with a function that approximates the relation between inputs (images) to the outputs (targets).

First then let's understand what a `DataBlock` is.

## Datablock

 A `DataBlock` is a template which tells the dataloader which will finally use this datablock, the following five things.

- **blocks:**  What are our inputs and outputs. For image classification, we are mapping an image to a category. So our input will be an `ImageBlock` and our output will be a `CategoryBlock`. For text classification, we would do a text-category mapping so it will have `TextBlock` as the input and `CategoryBlock` as the output. If we wish to have multiple inputs, we can do that by passing a tuple/list as the first element of the blocks tuple; same for output. For eg. if we wish to pass two images and see if they're same or different in some aspect, we can have `blocks = ([ImageBlock, ImageBlock], CategoryBlock)`. 

- **get_items**: We have to tell how to get the input. fastai has a prebuilt function `get_image_files` which gets all the images in a folder and it's sub folders (well, most common `.png`, `.jpg` formats etc. are considered. `.webp` etc. aren't) into a list.

- **splitter**: Any Machine Learning model given enough data can perfectly approximate a set of inputs to a set of outputs by memorizing it. However it will not perform well during runtime because it has not learnt the underlying concept and simply memorized the connection of input to output. So this basically specifies the way of splitting. For that fastai has it's own set of splitters which could be used. The most common is the `RandomSplitter` which takes a validation percentage and splits that much percentage into the validation set and remaining into the training set.

If you have a folder structure like this, i.e. your train and validation folders are explicitly defined in the structure shown below, you can choose to use the `GrandparentSplitter`

```yaml
cartoons
    - train
        - bugs bunny
        - donald duck
        - mickey mouse
        - shinchan
    - valid
        - bugs bunny
        - donald duck
        - mickey mouse
        - shinchan

```

![](https://i.imgur.com/rLvny3O.png)

By default the `GrandparentSplitter` assumes your training set is called `train` and your validation set is called `valid`. If that's not the case you can instantiate grandparent splitter by explicitly specifying the names of your folders respectively.

- **get_y**: This is to specify how do you want your model to decipher the category which should be assigned to the input in the ImageBlock. If you have a folder structure like above, you know that in the path of the items, the penultimate item in the path is always the class name, when that's the case you can use the `parent_label` function which is provided in fastai to specify that's where we need to pick the targets from.

- **item_tfms**:  When we download images from the web, we observe that not all of them are of the same resolution. Some are thumbnails, some are gigantic poster sized images, some are reasonably good resolution images etc. But when we feed our data to the model we need to make sure they're all the same size because we need to be able to batch them together and send it to the model for training. This is where this *item_tfms* comes in handy. If you want to manipulate your input before actually sending it to the model, you can do it here. We just use a `Resize` to basically ensure that what goes in is consistent. Also there's several ways to resize a picture like `squish, pad, random, crop etc.` but with this default method we do a `CenterCrop` for resizing the images.

Now, we're finally ready to define our datablock below.

```python
# Define the template to load the data
cartoons = DataBlock(
                      # Specify the kind of inputs and the kind of outputs
                      blocks=(ImageBlock, CategoryBlock), 
                      # Specify how to get the input
                      get_items=get_image_files, 
                      # Specify how to perform the train and validation splits
                      splitter=RandomSplitter(valid_pct=0.2, seed=42),
                      # Specify how to read the dependent label
                      get_y=parent_label,
                      # Before going on gpu, any transforms to be done should be specified here
                      item_tfms=Resize(256)
                     )
```

## Dataloader

This template can now be reused to create new templates or to define a `Dataloader`. A dataloader is an actual mechanism unlike a datablock which can be used to pass data to a model. We can create a dataloader out of the datablock defined above as

```python
dls = cartoons.dataloaders(path, bs = 32)
```

We need to specify the `batch size` which basically means how many images can the model see at once. By default it is selected but it's better to specify based on your own hardware requirements as larger batch sizes might not fit onto smaller gpus. So be sure to check that out.

Then we can see the data as it's loaded by using the `show_batch` method native to the Dataloader as follows.

![](https://i.imgur.com/kETijNE.png)

Deep Learning models are data hungry. The more data you throw at them, in most cases, the better they learn. Now that we have some data, we can create copies of that data by changing elements like the camera angle, brightness, warp etc. and this technique is called data augmentation i.e. virtually increasing the data by creating copies of it.

By copies, I mean, these copies are created only during run-time. `batch_tfms` short for batch transforms are carried out on a GPU (graphics card) and then the input is directly fed to the model at run time; so you don't have to augment data beforehand as you pass it to the model (you can totally do it if you want to but it's not necessary...)

In the above snippet we can see how augmentation of a same image can produce different images like flipping, random cropping, brightness manipulation etc. as shown in the second row.

## Model

Since we now have all the elements together, we can build our very own cartoon classifier by creating a model and passing it the data with the dataloaders created above. Doing that just takes 2 simple lines of code

```python
# Train a cnn model to classify these 4 different cartoons from one another
learn = cnn_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(8, freeze_epochs = 3)
```

![](https://i.imgur.com/F0btFPt.png)

We first create a `cnn_learner` using the `resnet18` model architecture which is a function having some weights that get updated based on a performance metric on the data relayed to it by `dls`.

We then use the `learn.fine_tune` method by freezing the epochs to some fixed number of epochs and then running by unfreezing for remaining epochs and training the model. On the fastbook slack, there's a really beautiful answer by [Henry here](https://wb-forum.slack.com/archives/C023P7TM1DK/p1624370132315400?thread_ts=1624263184.301900&cid=C023P7TM1DK) explaining the importance of fine tuning for a pretrained model. 

As we see by the end of 11 epochs, we're at 75% accuracy with just close to 200 images per class with so much variance in within the class. Not bad at all, isn't it :)? especially considering such few lines of code that we've written...

Now, it's time to evaluate the model i.e. identify where did it mispredict and which ones was our model struggling so much with?

## Model Evaluation

Confusion Matrix a very common tool which is utilized to assess the performance of classification models. It is a grid of values with rows having actual classes and columns having predicted classes and each cell is populated with the number of elements which actually belonged to an actual row class and were predicted as a predicted column class. [This article](https://medium.com/swlh/classification-metrics-and-the-confusion-matrix-3e35581c1153#89ea) nicely explains the confusion matrix concept in depth which I found helpful.

For our case, we can use fastai's interpretor to compute a confusion matrix as follows.

![](https://i.imgur.com/wiHcaMG.png)

We can see from the matrix that `shinchan` and `bugs bunny` have been very nicely classified looking rowwise since those are the actual values. However `donald duck` and `mickey mouse` seem very hard to classify since for those categories, few counts are along the respective diagonal and most of them are scattered across other classes. 

If we look at the images which are misclassified there's some semblance of shinchan in the second image (shinchan always wears a red shirt and khaki pants and mickey's attire has similar pants and boots respectively). In the first image, with mickey mouse there's another character who has long ears and is tall like bugs bunny. So our model has quite some scope for improvement but it's alright for now.

Finally, we can save this model weights to a file by calling an `.export` method which creates a file named export.pkl in same directory as the notebook. This contains all the necessary elements for making inference at the run time.

# The Final Mile: Building an app from the model

Now that we have made a classifier which can tell 4 different cartoons apart, let's make it usable i.e. deploy it to share it with friends and family. 

With the help of `ipywidgets` library, we can create interactive elements in the notebook itself.

All we need is a simple layout which has 

- An upload button which a user can click to upload an image
```python
from fastai.vision.widgets import *
# Create an upload button
upload_btn = widgets.FileUpload()
```

- An output widget to display the image which we have uploaded to the application
```python
# Create a widget to display the output
out_pl = widgets.Output()
```

- A label on which we can write the name of the predicted class and the confidence score for the prediction
```python
# Create a widget to display the output
out_pl = widgets.Output()
```

- A button which when clicked gets the prediction, clears the output, overlays the most recent input image on the output and updates the label to reflect the latest prediction.
```python
# Define a button which will classify on being clicked
btn_run = widgets.Button(description='Classify')
```

Now the button handling code can be written after all our widgets are defined above as follows and eventually register this function  with the button that we created for the `on_click` event.

```python
def on_click_classify(change):
    # Create an image out of the uploaded content
    img = PILImage.create(upload_btn.data[-1])
    
    # Clear the output associated with output widget
    out_pl.clear_output()
    
    # Display the image in the outpt widget
    with out_pl: display(img.to_thumb(128,128))
        
    # Make a prediction
    pred,pred_idx,probs = model.predict(img)
    
    # Show the prediction on the label
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)
```

Then we can host the notebook along with `requirements.txt`, `export.pkl` and an optional `README.md` file to github and create an app on binder which can be accessed by anyone for doing prediction. Here is a working example of the app that I created with the above model.

![](https://i.imgur.com/7M6FBKH.gif)

>**Note that when you make a repo for hosting these files, do not keep the files in a separate folder in the repo; let it be at the root level only otherwise the app will not build as expected on binder.**

# 10000 feet summary of the entire session

1. Create your own datasets using `search_images_ddg` or `search_images_bing`. It will help you get a feel for the entire end to end process of deep learning modelling.
2. `Datablock` is a template which defines how to load the data into the computer out of which `Dataloaders` are created. `Dataloaders` are the real deal which curate the data using the datablock in a form which is expected by the model by applying all the transforms (`item_tfms` on cpu and `batch_tfms` on gpu) to feed it to the model.
3. `cnn_learner` is a class which can be used to generate a basic learner object for image classification tasks and could be trained using `fit_one_cycle` method by freezing the body weights for some epochs and then releasing for subsequent epochs.
4. `ipywidgets` with the help of `voila` can help you create a simple notebook application for prediction; Uploading the project to github and using that link, we can create a sharable version of our application using `binder`.


If you're still with me, thanks for reading through. I appreciate your patience and hope this post helped you get started with the fastai library for image classification. I would be glad to [connect with you on Twitter](https://twitter.com/ElisonSherton) or in the discussion below should you have any doubts or suggestions for improvements :)

# References

1. [fastbook chapter 2](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)
2. [Session link](https://www.youtube.com/watch?v=rmOqCO7c8pw)
3. [Understanding confusion matrix](https://medium.com/swlh/classification-metrics-and-the-confusion-matrix-3e35581c1153#89ea)
4. [Github code for the application created in the post](https://github.com/ElisonSherton/fastbook_sessions/tree/master/ch2Production)