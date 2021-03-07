---
layout: page
title: All Posts
---
<ul>
{% for post in site.posts %}
    <li>
        <span><a href = "{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a> ({{ post.date | date_to_string }})</span>
    </li>
{% endfor %}
</ul>