---
layout: page
title: About
permalink: /about/
published: true
---

<div class="page" markdown="1">

{% capture page_subtitle %}
<img
    class="me"
    alt="{{ author.name }}"
    src="{{ site.author.photo | relative_url }}"
    srcset="{{ site.author.photo2x | relative_url }} 2x"
/>
{% endcapture %}

{% include page/title.html title=page.title subtitle=page_subtitle %}

## littlepure

Hi, I'm littlepure, a newcomer of computer vision. I like to create, experience interesting things. I always believe in a word "one is born to do a big thing". And I wish I can live a life not for nothing.😎

<img src="resources/Luffy_vs_Ulti.png" align="middle">

</div>
