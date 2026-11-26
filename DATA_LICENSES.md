# Data licences and attribution

The [MIT licence](LICENSE) applies to BauLens code. The dataset records below were verified on
2026-11-26.

## Concrete Compressive Strength

- Source and dataset identifier are the [UCI Machine Learning Repository, ID 165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
  and [DOI 10.24432/C5PK67](https://doi.org/10.24432/C5PK67).
- Licence is [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  as stated on the UCI record.
- BauLens downloads the 1,030-row workbook for compressive-strength regression. The
  redistributed `tests/fixtures/concrete_sample.csv` is a 60-row subset sampled with
  `random_state=42` and exported in source row order. Column names were converted to
  snake_case. Measurements retain the original units and values.

Yeh, I.-C. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository.
[10.24432/C5PK67](https://doi.org/10.24432/C5PK67).

```bibtex
@misc{yeh1998concrete,
  author = {Yeh, I-Cheng},
  title = {Concrete Compressive Strength},
  year = {1998},
  publisher = {UCI Machine Learning Repository},
  doi = {10.24432/C5PK67},
  url = {https://doi.org/10.24432/C5PK67}
}
```

## Concrete Crack Images for Classification

- Source and dataset identifier are [Mendeley Data, version 2](https://data.mendeley.com/datasets/5y9wdsg2zt/2)
  and [DOI 10.17632/5y9wdsg2zt.2](https://doi.org/10.17632/5y9wdsg2zt.2).
- Licence is [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  as stated on the Mendeley Data record.
- BauLens downloads 40,000 images for crack classification and keeps the archive and real
  images local. Test images are generated from noise and drawn lines.

Özgenel, Ç. F. (2019). *Concrete Crack Images for Classification* [Dataset, version 2].
Mendeley Data. [10.17632/5y9wdsg2zt.2](https://doi.org/10.17632/5y9wdsg2zt.2).

The dataset record also requests citation of the following paper.

Özgenel, Ç. F., & Gönenç Sorguç, A. (2018). Performance comparison of pretrained convolutional
neural networks on crack detection in buildings. *Proceedings of the 35th International
Symposium on Automation and Robotics in Construction*, 693–700.
[10.22260/ISARC2018/0094](https://doi.org/10.22260/ISARC2018/0094).

The publisher record additionally lists this paper among the references to cite.

Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). Road crack detection using deep
convolutional neural network. *2016 IEEE International Conference on Image Processing (ICIP)*,
3708–3712. [10.1109/ICIP.2016.7533052](https://doi.org/10.1109/ICIP.2016.7533052).

```bibtex
@misc{ozgenel2019cracks,
  author = {Özgenel, Çağlar Fırat},
  title = {Concrete Crack Images for Classification},
  year = {2019},
  publisher = {Mendeley Data},
  version = {2},
  doi = {10.17632/5y9wdsg2zt.2},
  url = {https://doi.org/10.17632/5y9wdsg2zt.2}
}

@inproceedings{ozgenel2018performance,
  author = {Özgenel, Çağlar Fırat and Gönenç Sorguç, Arzu},
  title = {Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings},
  booktitle = {Proceedings of the 35th International Symposium on Automation and Robotics in Construction},
  year = {2018},
  pages = {693--700},
  doi = {10.22260/ISARC2018/0094},
  url = {https://doi.org/10.22260/ISARC2018/0094}
}
```

## SDNET2018

Optional dataset for future external evaluation. BauLens currently uses the Mendeley images
for crack classification. The [Utah State University dataset record](https://digitalcommons.usu.edu/all_datasets/48/)
lists [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). *SDNET2018: A concrete crack image dataset
for machine learning applications* [Dataset]. Utah State University.
[10.15142/T3TD19](https://doi.org/10.15142/T3TD19).

```bibtex
@misc{maguire2018sdnet,
  author = {Maguire, Marc and Dorafshan, Sattar and Thomas, Robert J.},
  title = {{SDNET2018}: A Concrete Crack Image Dataset for Machine Learning Applications},
  year = {2018},
  publisher = {Utah State University},
  doi = {10.15142/T3TD19},
  url = {https://doi.org/10.15142/T3TD19}
}
```
