---
layout: post
title: Author Identification using Naive Bayes Algorithm — Model Building
published: true
categories: ['Machine Learning']
---

This post is the second in a series of three posts which aim to build a naive bayes classification algorithm in an end to end fashion. In the [previous post](https://elisonsherton.github.io//machine%20learning/2021/01/01/naive-bayes-author-identification-1.html), we looked at curating the data for this problem: We scraped dialogues for both Captain America and Iron Man from the transcripts of Captain America: The First Avenger, Iron Man and Avengers: Endgame respectively. We then processed them to remove all action/scene description cues and only retain the spoken word by both our characters and created a dataframe out of the same and saved them as csv files.

![](https://miro.medium.com/max/1920/1*j3tuPnOkDRrt39L7gOcwzg.jpeg)

We shall now focus on building a Naive Bayes Classifier along with the working for the same from scratch. In this post, we will focus on

- Understanding Conditional Probability

- Understanding Bayes Theorem

- Naive Bayes for author identification

- Evaluating the Classification Model

Without any further ado, let’s get started!

<hr>