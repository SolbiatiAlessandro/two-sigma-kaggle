This are the kernels I am going through with some comments

top 10 by gain
--
returnsOpenPrevMktres10 : 25540.090746879578
assetName_mean_close : 5334.778017044067
returnsOpenPrevRaw1 : 3812.769549369812
assetName_mean_open : 2712.5761909484863
close : 1865.5509052276611
close_to_open : 1781.2829246520996
returnsClosePrevMktres1 : 1490.8566799163818
open : 1401.5982103347778
returnsOpenPrevMktres1 : 754.6836042404175
returnsClosePrevRaw1 : 295.4302005767822

top 10 by split
--
returnsOpenPrevMktres10 : 52
assetName_mean_close : 52
returnsOpenPrevRaw1 : 48
open : 46
assetName_mean_open : 39
close : 34
returnsOpenPrevMktres1 : 31
close_to_open : 28
returnsClosePrevMktres1 : 25
returnsClosePrevRaw1 : 16

- [1] [Getting Started](https://www.kaggle.com/dster/two-sigma-news-official-getting-started-kernel)
Introductory kernel that explains how to use APIs

- [2] [EDA, feature engineering and everyhing](https://www.kaggle.com/artgor/eda-feature-engineering-and-everything)
This is the most upvoted Kernel, I am starting from here to understand data and how to approach the task

***Key takeaways***:

	- Market Data: (errors) there are some outliers that are cleaned in the beginning of the kernel, taking data only after 2010 
	- News Data: urgency is 1 or 3, mostly 15-25chars length with some big outliers, most news are tagless 
	- Model: binary classifier

- [3] [Bird-Eye View + NN Approach](https://www.kaggle.com/ashishpatel26/bird-eye-view-of-two-sigma-nn-approach)
This has a lot of nice plots, and then basically throw everything (not the second dataset tough) into an NN. Is mentioned in [2]

- [4] [Guowenrui NN Kernel](https://www.kaggle.com/guowenrui/market-nn-if-you-like-you-can-use-it-and-upvote)
This is a basic NN kernel, useful as a reference. The guy is actually pretty strong and is scoring top 100 right now, I contacted him and we might consider to form a team in the future.

- [5] [eda script 67]()
This is the best public script up to now, scores 0.69. Let's explore the top features. There are two lgb models, with averaging.

***FEATURE EXPLORATION***

gbm_1

top10 by gain
--
returnsClosePrevRaw10_lag_7_mean : 56124.36911582947
returnsClosePrevMktres10_lag_7_min : 18476.50610923767
returnsClosePrevMktres10_lag_7_mean : 13604.055717468262
assetCodeT : 12110.821272850037
returnsClosePrevRaw10_lag_14_max : 10597.433120727539
returnsClosePrevRaw10_lag_14_mean : 10578.360836029053
returnsClosePrevRaw10_lag_14_min : 9953.875637054443
returnsClosePrevMktres10_lag_14_max : 9637.041584968567
returnsClosePrevMktres10_lag_14_mean : 9315.482937812805
returnsClosePrevRaw10_lag_7_min : 7727.059662818909

top10 by split
--
assetCodeT : 743
returnsClosePrevRaw10_lag_14_min : 553
returnsClosePrevMktres10_lag_14_max : 516
returnsClosePrevRaw10_lag_14_max : 504
returnsClosePrevMktres10_lag_14_min : 449
volume : 426
returnsClosePrevRaw10_lag_14_mean : 407
returnsClosePrevRaw10_lag_7_max : 360
returnsClosePrevRaw10_lag_7_min : 347
returnsClosePrevMktres10_lag_7_max : 328

gbm_2

top10 by gain
--
returnsClosePrevRaw10_lag_7_mean : 61099.98406600952
returnsClosePrevMktres10_lag_7_min : 20410.298405647278
assetCodeT : 15830.2543592453
returnsClosePrevMktres10_lag_7_mean : 15244.119355201721
returnsClosePrevRaw10_lag_14_max : 13072.144352912903
returnsClosePrevRaw10_lag_14_min : 12746.133717536926
returnsClosePrevRaw10_lag_14_mean : 12674.183384895325
returnsClosePrevMktres10_lag_14_max : 11672.46259689331
returnsClosePrevMktres10_lag_14_mean : 10987.076703071594
returnsClosePrevMktres10_lag_14_min : 9644.152087211609

top10 by split
--
assetCodeT : 999
returnsClosePrevRaw10_lag_14_min : 736
returnsClosePrevRaw10_lag_14_max : 658
returnsClosePrevMktres10_lag_14_max : 643
volume : 589
returnsClosePrevMktres10_lag_14_min : 586
returnsClosePrevRaw10_lag_14_mean : 539
returnsClosePrevMktres10_lag_7_max : 451
returnsClosePrevRaw10_lag_7_min : 445
returnsClosePrevRaw10_lag_7_max : 430

Looks like lagged features 7 and 14 of max,min,mean of 10 days before price are the best predictive.
