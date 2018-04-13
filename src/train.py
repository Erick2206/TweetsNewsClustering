from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from gensim.models.doc2vec import Doc2Vec

n_clusters=5000
labels=range(45448)

def loadVectors():
    fname='doc2vec.model'
    model = Doc2Vec.load(fname)
    return model

def trainKMeans(vec):
    textVect = vec.docvecs.doctag_syn0
    print "Initiating K Means Clustering-"
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=True)

    km.fit(textVect)
    print "Results for K Means Clustering:"
    print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_)
    print "Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_)
    print "V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_)
    print "Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_)
    print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(textVect, km.labels_, sample_size=1000)
    print ""
    print ""

def trainAgglo(vec):
    textVect = vec.docvecs.doctag_syn0
    print "Initiating Agglomerative Clustering-"
    agglo = AgglomerativeClustering(n_clusters=n_clusters,
                                        linkage="average", affinity="cosine")
    agglo.fit(textVect)
    print "Results for Agglomerative Clustering:"
    print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, agglo.labels_)
    print "Completeness: %0.3f" % metrics.completeness_score(labels, agglo.labels_)
    print "V-measure: %0.3f" % metrics.v_measure_score(labels, agglo.labels_)
    print "Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, agglo.labels_)
    print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(textVect, agglo.labels_, sample_size=1000)
    print ""
    print ""


def main():
    model=loadVectors()
    trainKMeans(model)
    trainAgglo(model)


if __name__=='__main__':
    main()
