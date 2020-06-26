package main

import (
	"fmt"
	"github.com/pa-m/sklearn/cluster"
	"github.com/pa-m/sklearn/datasets"
	"time"
)

func main() {
	start := time.Now()
	_ = start
	kmeansBlobs()
	fmt.Printf("elapsed %s s\n", time.Since(start))
}

func kmeansBlobs(){
	//datasets.MakeBlobs(&datasets.MakeBlobsConfig{NSamples: 1000000,
	//	Centers:    3,
	//	ClusterStd: 0.5})

	X,Y := datasets.MakeBlobs(&datasets.MakeBlobsConfig{NSamples: 10000,
		Centers: 10,
		ClusterStd: 0.5})
	kmeans := &cluster.KMeans{NClusters: 10}
	start := time.Now()
	_ = start
	kmeans.Fit(X, nil)
	kmeans.Predict(X, Y)
	fmt.Printf("elapsed %s s\n", time.Since(start))
}
