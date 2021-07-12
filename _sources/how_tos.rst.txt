.. _how_to:

========
How-To
========

This section is a list of miscellaneous task-oriented tutorials for specific usages of ``adfluo``

Dealing with large datasets
+++++++++++++++++++++++++++

This tutorial is aimed relevant if you're planning on using ``adfluo`` to extract feature from large datasets and extraction pipelines that:

- have samples that require a large amount of memory to be processed
  (e.g. : very long audio recordings)
- have so many small samples that loading all of them at once might fill
  your computer's memory (e.g: a myriad of small images)
- have implemented processors that output intermediate data that might also be very large
- some or all of the above

What follows is some advice on best to use ``adfluo`` to prevent your feature computation code from
eating all of your RAM.

Make data loading as lazy as possible
-------------------------------------

TODO

Use sample-wise extraction order
--------------------------------

TODO

Active streaming for the output's storage
-----------------------------------------

TODO


Picking the right storage format
++++++++++++++++++++++++++++++++

TODO

Sharing your extraction pipelines
+++++++++++++++++++++++++++++++++

pass