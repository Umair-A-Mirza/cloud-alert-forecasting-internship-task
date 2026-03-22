# cloud-alert-forecasting-internship-task

A test task for an internship screening on the topic of Predictive Alerting for Cloud Metrics. The task was to implement a model to predict whether an incident were to occur in the next T time steps given a previous window W of data. A sliding window approach was requested.

### Dataset Choice

The chosen dataset is **The Numenta Anomaly Benchmark (NAB)**, which focuses on anomaly detection in various different cases, such as internet traffic, tweets, but notably also AWS CloudWatch logs, which will be the focus of this implementation.

It is important to note that for the internship itself, I do not expect just one data source for cloud metrics, rather, it will require some consolidation of data from various different distributed cloud solutions in order to capture a more clear understanding. This requires a great degree of feature engineering and in-depth knowledge of the dataset. Such datasets are hard to find curated and available for free online, and the **Numenta** dataset is known more for the ease of access and simplicity.

Therefore, although this implementation still focuses on a similar task, it is important to recognize that the dataset may not accurately reflect to the same degree what will be expected from the internship.

The AWSCloudWatch data provided by **Numenta** contains multiple CSVs, and the following will be used:
- **CPU_UTIL_1** - ec2_cpu_utilization_5f5533.csv
- **CPU_UTIL_2** - ec2_cpu_utilization_53ea38.csv

### Exploratory Data Analysis (EDA)

In any problem, it is important to first explore the provided data. In this case, whilst it may seem simple that we have just one target value in the selected datasets, it will likely be more challenging to determine anomalous behavior solely from a single feature's lagged/historical values. Nevertheless, a core element of the EDA in this case will be to determine how informative previous values of the target are, and whether we can find any insightful patterns or qualities about the data.

From experience with cloud metrics, they are typically noisy and sensitive to time-of-day, especially in cases like Lambdas being invoked, which depend on business activity. In this case, we are interested more in rare fluctuations or movements of the target variable. This involves determining appropriate window sizes for a history, the nature of the time-series itself (relatively stationary or very unstable), the relative scale of the target, influence of seasonalities, and more.

This is handled by the following pipeline, each step focused on providing an overview of information that can be useful in a thorough analysis:
1. **Basic Summary** - A summary table is then computed with the number of rows, start and end time, median and modal sampling interval, number of missing timestamps, number of duplicate timestamps, minimum, maximum, mean, and standard deviation of the metric.
    - This is a general step in any EDA pipeline, but specifically for sliding-window models, we want to ensure the sequence is sampled consistently.

2. **Time-Series Visualization with Anomaly Overlays** - When training a model for anomaly detection, we recognize that anomalous behavior is usually a small subset of the overall stream of data.
    - The major advantage of this plot is to help detect whether anomalies are isolated spikes or longer abnormal phases, and if they coincide with seasonal patterns, which can be useful if we construct features related to seasonality.

3. **Distribution Analysis** - A histogram and boxplot are used to characterize the distribution of values in order to identify skewness, heavy tails, and outliers.
    - This helps identify the typical operational conditions for the service.

4. **Rolling Statistics** - Rolling mean and rolling standard deviation are plotted over time, which can help identify how the baseline changes as well as outline any volatility. 
    - This can be helpful, as an early assumption is that anomalous behavior should coincide with periods of higher volatility.

5. **Autocorrelation Analysis** - Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots are used to inspect temporal dependence.
    - Temporal dependence is critical for establishing if relatively recent history is predictive of near-future behavior, and whether correlations decay/persist. It can also reveal seasonalities that we should be aware of in the modeling.

6. **Seasonal Pattern Analysis** - In addition to the Autocorrelation Analysis, we can check for basic seasonality in these datasets by some rudimentary plots of mean value per hour and per day.
    - In business contexts, we may be able to associate some activity/load based on normal operating hours. Behavior in cloud metrics is certainly time-dependent.

7. **Lag Plots** - Lag plots are used to visualize the relationship between the target and its lagged values, for multiple lag. 
    - In this project, points are also colored according to whether the current point, the lagged point, or both lie inside anomaly windows.
    - These plots will help outline temporal dependence, possible seasonality at selected lags, and some patterns with regards to anomalous behavior.

8. **Fast Fourier Transforms** - FFT is an algorithm to compute Discrete Fourier Transform and its inverse to transform the signal form time domain to frequency domain.
    - The primary use is to identify dominant periods in order to determine a reasonable window size for selection.

9. **Anomaly Statistics** - Because the ground truth is defined by anomaly windows, we should note the number of anomaly windows, total anomaly duration, average anomaly duration, number of anomalous points, and fraction of points inside anomaly windows.
    - This will help us understand how rare anomalous behavior is in the dataset, and whether anomalous behavior is short-lived or persists for longer periods of time.

### Relevance of Stationarity

If we were to forecast values with classical approaches, stationarity almost certainly becomes a requirement or implicit assumption with respect to many modeling approaches. In this case, our task is to predict whether an anomaly occurs in the next H time steps when given a time window W. This is a **supervised classification** problem with binary labels, which does not require stationarity with the same strength as other use-cases (e.g. forecasting a value).

Non-stationarity usually indicates elements such as trend changes or drift, which can be common for enterprise cloud data. Increased traffic as operations grow or the deployment of new services can both contribute to dramatic changes in observed cloud metrics, but neither are anomalous. Yet, both of these benign cases will possibly leave a series as non-stationary. Even if a series is non-stationary, the model needs to predict based on provided data in a local window, and the prediction target is event occurrence. If there is not a reasonable manner in which we can approximately make the global data generating process stationary, then instead our attention should be put towards modeling choices which can help generalize well from local context and learn short-term behavior (e.g. volatility, temporal dynamics) better.

However, some degree of stationarity still remains relevant for the model to establish a logical baseline to detect abrupt spikes, which are likely periods of abnormal activity.

