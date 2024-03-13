extern crate rand;

use rand::Rng;
use std::ptr;
use std::os::raw::c_int;
use std::os::raw::c_uint;
use std::os::raw::c_void;
use std::option::Option;
use std::slice;
use num_traits::{Num, ToPrimitive};

use crate::rand::distributions::Distribution;

pub type Pointer = *mut c_void;
pub type KmeansResult = c_uint;
pub const KMEANS_OK: KmeansResult = 0;
pub const KMEANS_EXCEEDED_MAX_ITERATIONS: KmeansResult = 1;
pub const KMEANS_ERROR: KmeansResult = 2;
pub type KmeansDistanceMethod = Option<unsafe extern "C" fn(a: Pointer, b: Pointer, l: usize) -> f64>;
pub type KmeansCentroidMethod = Option<
    unsafe extern "C" fn(
        objs: *const Pointer,
        clusters: *const c_int,
        num_objs: usize,
        cluster: c_int,
        centroid: Pointer,
        l: usize,
    ),
>;
#[repr(C)]
pub struct kmeans_config {
    pub distance_method: KmeansDistanceMethod,
    pub centroid_method: KmeansCentroidMethod,
    pub objs: *mut Pointer,
    pub num_objs: usize,
    pub centers: *mut Pointer,
    pub l: usize,
    pub k: c_uint,
    pub max_iterations: c_uint,
    pub total_iterations: c_uint,
    pub clusters: *mut c_int,
}

#[link(name = "ckmeans")]
extern "C" {
    pub fn kmeans(config: *mut kmeans_config) -> KmeansResult;
}

unsafe extern "C" fn calc_distance_f64(
    a: Pointer,
    b: Pointer,
    l: usize,
) -> f64 {
    let point_a = slice::from_raw_parts(a as *const f64, l);
    let point_b = slice::from_raw_parts(b as *const f64, l);

    let distance_squared: f64 = point_a
        .iter()
        .zip(point_b.iter())
        .map(|(&x, &y)| ((x as f64) - (y as f64)).powi(2))
        .sum();
    
    // println!("PA: {:?}, PB: {:?}, Dist: {:?}", a, b, distance_squared);

    distance_squared
}

unsafe extern "C" fn calc_distance_u8(
    a: Pointer,
    b: Pointer,
    l: usize,
) -> f64 {
    let point_a = slice::from_raw_parts(a as *const u8, l);
    let point_b = slice::from_raw_parts(b as *const f64, l);

    let distance_squared: f64 = point_a
        .iter()
        .zip(point_b.iter())
        .map(|(&x, &y)| ((x as u8).to_f64().unwrap() - (y as f64)).powi(2))
        .sum();
    
    // println!("PA: {:?}, PB: {:?}, Dist: {:?}", a, b, distance_squared);

    distance_squared
}

unsafe extern "C" fn calc_centroid_f64(
    objs: *const Pointer,
    clusters: *const c_int,
    num_objs: usize,
    cluster: c_int,
    centroid: Pointer,
    l: usize,
) {
    let mut sum_by_dimension: Vec<f64> = vec![0.0; l];
    let mut count = 0;

    for i in 0..num_objs {
        if *clusters.add(i) == cluster {
            let point = &*(objs.add(i) as *const *const f64);
            for j in 0..l {
                sum_by_dimension[j] += *(point.add(j) as *const f64) as f64;
            }
            count += 1;
        }
    }

    if count > 0 {
        for j in 0..l {
            *(((centroid as *mut Vec<f64>).add(0)) as *mut f64)
                .add(j) = sum_by_dimension[j] / count as f64;
        }
    }
}

unsafe extern "C" fn calc_centroid_u8(
    objs: *const Pointer,
    clusters: *const c_int,
    num_objs: usize,
    cluster: c_int,
    centroid: Pointer,
    l: usize,
) {
    let mut sum_by_dimension: Vec<f64> = vec![0.0; l];
    let mut count = 0;

    for i in 0..num_objs {
        if *clusters.add(i) == cluster {
            let point = &*(objs.add(i) as *const *const u8);
            for j in 0..l {
                sum_by_dimension[j] += (*(point.add(j) as *const u8) as u8).to_f64().unwrap();
            }
            count += 1;
        }
    }

    if count > 0 {
        for j in 0..l {
            *(((centroid as *mut Vec<f64>).add(0)) as *mut f64)
                .add(j) = sum_by_dimension[j] / count as f64;
        }
    }
}


fn cluster_generic<T: Num + ToPrimitive>(
    data: &Vec<Vec<T>>,
    cluster_count: u32,
    max_iteration_count: u32,
    distance_method: KmeansDistanceMethod,
    centroid_method: KmeansCentroidMethod,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    let mut config = kmeans_config {
        distance_method: distance_method,
        centroid_method: centroid_method,
        objs: ptr::null_mut(),
        num_objs: data.len(),
        centers: ptr::null_mut(),
        l: data[0].len(),
        k: cluster_count,
        max_iterations: max_iteration_count,
        total_iterations: 0,
        clusters: ptr::null_mut(),
    };

    let mut rng = rand::thread_rng();

    // Allocate memory for objs, centers, and clusters
    let mut objs: Vec<Pointer> = data
        .iter()
        .map(|point| point.as_ptr() as Pointer)
        .collect();
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(cluster_count as usize);
    for i in 0..cluster_count {
        let mut center: Vec<f64> = Vec::with_capacity(data[0].len());
        for j in 0..data[0].len() {
            center.push(data[i as usize][j].to_f64().unwrap());
        }
        centers.push(center);
    }

    let mut center_pointers: Vec<Pointer> = (0..cluster_count)
        .map(|i| centers[i as usize].as_ptr() as Pointer)
        .collect();
    let mut clusters: Vec<c_int> = Vec::with_capacity(data.len());
    for _ in 0..data.len() {
        clusters.push(rng.gen_range(0..cluster_count) as c_int);
    }

    // Assign the allocated memory to the kmeans_config struct
    config.objs = objs.as_mut_ptr();
    config.centers = center_pointers.as_mut_ptr();
    config.clusters = clusters.as_mut_ptr();

    // Call the kmeans function
    let result = unsafe { kmeans(&mut config) };

    let converted_clusters: Vec<u32> = clusters.into_iter().map(|x| x as u32).collect();

    (centers, converted_clusters)
}


pub fn cluster_f64(
    data: &Vec<Vec<f64>>,
    cluster_count: u32,
    max_iteration_count: u32,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    cluster_generic::<f64>(
        data,
        cluster_count,
        max_iteration_count,
        Some(calc_distance_f64),
        Some(calc_centroid_f64),
    )
}

pub fn cluster_u8(
    data: &Vec<Vec<u8>>,
    cluster_count: u32,
    max_iteration_count: u32,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    cluster_generic::<u8>(
        data,
        cluster_count,
        max_iteration_count,
        Some(calc_distance_u8),
        Some(calc_centroid_u8),
    )
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_clustering() {
        println!("Testing with f64 values.");

        let data_f64: Vec<Vec<f64>> = vec![
            vec![0.0f64, 0.1f64, 0.0f64, 0.0f64],
            vec![0.1f64, 0.1f64, 0.0f64, 0.0f64],
            vec![0.0f64, 0.1f64, 0.1f64, 0.0f64],
            vec![2.0f64, 2.1f64, 2.0f64, 0.0f64],
            vec![2.1f64, 2.1f64, 2.0f64, 0.0f64],
            vec![2.0f64, 2.1f64, 2.1f64, 0.0f64],
        ];
        // let data = generate_random_data_points(10);
        println!("Data: {:?}", data_f64);
        println!("Data Length: {:?}", data_f64.len());
        let (centers_f64, clusters_f64) = cluster_f64(&data_f64, 2 as u32, 100 as u32);

        println!("Kmeans centers: {:?}", centers_f64);
        println!("Kmeans clusters: {:?}", clusters_f64);

        println!("Testing with u8 values.");

        let data_u8: Vec<Vec<u8>> = vec![
            vec![0u8, 1u8, 0u8, 0u8],
            vec![1u8, 1u8, 0u8, 0u8],
            vec![0u8, 1u8, 1u8, 0u8],
            vec![20u8, 21u8, 20u8, 0u8],
            vec![21u8, 21u8, 20u8, 0u8],
            vec![20u8, 21u8, 21u8, 0u8],
        ];
        // let data = generate_random_data_points(10);
        println!("Data: {:?}", data_u8);
        println!("Data Length: {:?}", data_u8.len());
        let (centers_u8, clusters_u8) = cluster_u8(&data_u8, 2 as u32, 100 as u32);

        println!("Kmeans centers: {:?}", centers_u8);
        println!("Kmeans clusters: {:?}", clusters_u8);
    }
}
