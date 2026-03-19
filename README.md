# cloud-alert-forecasting-internship-task

A test task for an internship offered by Jetbrains on the topic of Predictive Alerting for Cloud Metrics. The task was to implement a model to predict whether an incident were to occur in the next T time steps given a previous window W of data. A sliding window approach was requested.

### Dataset Choice

The chosen dataset is **The Numenta Anomaly Benchmark (NAB)**, which focuses on anomaly detection in various different cases, such as internet traffic, tweets, but notably also AWS CloudWatch logs, which will be the focus of this implementation.

It is important to note that for the internship itself, I do not expect just one data source for cloud metrics, rather, it will require some consolidation of data from various different distributed cloud solutions in order to capture a more clear understanding. This requires a great degree of feature engineering and in-depth knowledge of the dataset. Such datasets are hard to find curated and available for free online, and the **Numenta** dataset is known more for the ease of access and simplicity.

Therefore, although this implementation still focuses on a similar task, it is important to recognize that the dataset may not accurately reflect to the same degree what will be expected from the internship.

The AWSCloudWatch data provided by **Numenta** contains multiple CSVs, and the following will be used:
- **CPU_UTIL_1** - ec2_cpu_utilization_5f5533.csv
- **CPU_UTIL_2** - ec2_cpu_utilization_53ea38.csv
- **DISK_WRITE_1** - ec2_disk_write_bytes_c0d644.csv
- **DISK_WRITE_2** - ec2_disk_write_bytes_1ef3de.csv
- **NETWORK_IN_1** - ec2_network_in_5abac7.csv
- **NETWORK_IN_2** - ec2_network_in_257a54.csv
- **REQUEST_COUNT** - elb_request_count_8c0756.csv

The files above include different types of operations and data streams, as the prospective solution should be adaptable.