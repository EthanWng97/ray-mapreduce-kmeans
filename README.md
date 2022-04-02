# ray-mapreduce-kmeans
![ray-mapreduce](https://miro.medium.com/max/1400/1*2omU7XHeJUWgZ3kleRJ-OA.png)

## Full document
> [Medium](https://medium.com/navepnow/ray-supported-high-performance-distributed-clustering-algorithm-46389d422802)

## Prerequisites

* Python3

## Install
    pip install -r requirements.txt

## Usage
    python3 main.py -d working-dir -f input-file -s number-of-sample -k number-of-clusters -n number-of-iteration -m number-of-mappers -t number-of-tasks

* `working-dir`: working directory(also directory of check-in dataset)
* `input-file`: file name of dataset
* `number-of-sample`: number of samples you want to cluster
* `number-of-clusters`: number of clusters
* `number-of-iteration`: max iteration for clustering
* `number-of-mappers`: mappers in MapReduce
* `number-of-tasks`: tasks in MapReduce

## Run tests
    python3 main.py -d /Users/evan-mac/checkin -f loc-gowalla_totalCheckins.txt -s 50000 -k 20 -n 10 -m 5 -t 2

## Author

👤 **Evan**

* Twitter: [@NavePnow](https://twitter.com/NavePnow)
* Github: [@NavePnow](https://github.com/NavePnow)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!
Feel free to check [issues page](https://github.com/NavePnow/ray-mapreduce-kmeans/issues).

## 💰 Show your support

Give a ⭐️ if this project helped you!

| PayPal                                                                                                                                                                       | Patron                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=DSZJCN4ZUEW74&currency_code=USD&source=url) |   <a href="https://www.patreon.com/NavePnow"> <img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="160"> </a>

## 📖 Reference
> 1. Lloyd, Stuart P. (1957). "Least square quantization in PCM". IEEE Transactions on Information Theory, VOL. IT-28, NO. 2, March 1982, pp. 129–137.
> 2. Arthur, D.; Vassilvitskii, S. (2007). "k-means++: the advantages of careful seeding". Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms. Society for Industrial and Applied Mathematics Philadelphia, PA, USA. pp. 1027–1035.
> 3. B. Bahmani, B. Moseley, A. Vattani, R. Kumar, S. Vassilvitskii "Scalable K-means++" 2012 Proceedings of the VLDB Endowment.
> 4. Elkan, Charles (2003). "Using the triangle inequality to accelerate kmeans" (PDF). Proceedings of the Twentieth International Conference on Machine Learning (ICML).
> 5. "MapReduce Tutorial". Apache Hadoop. Retrieved 3 July 2019.
> 6. Marozzo, F.; Talia, D.; Trunfio, P. (2012). "P2P-MapReduce: Parallel data processing in dynamic Cloud environments" (PDF). Journal of Computer and System Sciences. 78 (5): 13821402.
> 7. "Example: Count word occurrences". Google Research. Retrieved September 18, 2013.
> 8. Berlińska, Joanna; Drozdowski, Maciej (2010-12-01). "Scheduling divisible MapReduce computations". Journal of Parallel and Distributed Computing. 71 (3): 450–459.
> 9. Philipp Moritz et al. 2018. Ray: A Distributed Framework for Emerging AI Applications. In 13th USENIX Symposium on OSDI '18. 561-577.
> 10. M. Zaharia, M. Chowdhury, M. J. Franklin, S. Shenker, and I. Stoica. Spark: cluster computing with working sets. In Proceedings of the 2nd USENIX conference on Hot topics in cloud computing, HotCloud'10, pages 10--10, Berkeley, CA, USA, 2010. USENIX Association.
## 🙏 Acknowledgments
* Ray Community

## 📝 License

---
Copyright © 2020 [Evan](https://github.com/NavePnow).
This project is MIT licensed.
