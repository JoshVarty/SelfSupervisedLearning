## Effect of Pretext Dataset Size on Downstream Performance

**Hypothesis:** By increasing the size of our pretext dataset we can improve downstream performance.

**Result:** True, but perhaps with saturating improvements.

**Methodology:**

Image inpainting pretext task.

Using four subsets of the ImageWang dataset:

- `/train` that has a corresponding class in `/val`
  - `1,275` images
- All `/train` data
  - `14,669` images
- All `/train` data + all `/unsup` data
  - `22,419` images
- All `/train` data + all `/unsup` data + all `/val` data
  - `26,348` images
  
Results: 

Random: **52.3%**

Partial /train: **53.1%**

All /train: **55.7%**

All /train and /unsup : **56.2%**

All /train,/unsup and /val : **56.3%**


![test](https://i.imgur.com/ZuuAygJ.png)
