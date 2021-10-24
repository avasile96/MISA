def kmeans_init(img, k):
    means, labels = kmeans2(img, k)
    try:
        means = np.array(means) #mean
        cov = np.array([np.cov(img[labels == i].T) for i in range(k)]) #covariance
        ids = set(labels) #labels
        pis = np.array([np.sum([labels == i]) / len(labels) for i in ids]) # general probability of a pixel belonging to a cluster
    except Exception as ex:
        pass
    return means, cov, pis