extern crate rand;

use rand::Rng;
use std::ptr;
use std::os::raw::c_int;
use std::os::raw::c_uint;
use std::os::raw::c_void;
use std::option::Option;

use crate::rand::distributions::Distribution;

pub type Pointer = *mut c_void;
pub type KmeansResult = c_uint;
pub const KMEANS_OK: KmeansResult = 0;
pub const KMEANS_EXCEEDED_MAX_ITERATIONS: KmeansResult = 1;
pub const KMEANS_ERROR: KmeansResult = 2;
pub type KmeansDistanceMethod = Option<unsafe extern "C" fn(a: Pointer, b: Pointer) -> f64>;
pub type KmeansCentroidMethod = Option<
    unsafe extern "C" fn(
        objs: *const Pointer,
        clusters: *const c_int,
        num_objs: usize,
        cluster: c_int,
        centroid: Pointer,
    ),
>;
#[repr(C)]
pub struct kmeans_config {
    pub distance_method: KmeansDistanceMethod,
    pub centroid_method: KmeansCentroidMethod,
    pub objs: *mut Pointer,
    pub num_objs: usize,
    pub centers: *mut Pointer,
    pub k: c_uint,
    pub max_iterations: c_uint,
    pub total_iterations: c_uint,
    pub clusters: *mut c_int,
}

#[link(name = "ckmeans")]
extern "C" {
    pub fn kmeans(config: *mut kmeans_config) -> KmeansResult;
}

unsafe extern "C" fn calc_distance(a: Pointer, b: Pointer) -> f64 {
    let point_a = &*(a as *const Vec<f64>);
    let point_b = &*(b as *const Vec<f64>);

    let distance_squared: f64 = point_a
        .iter()
        .zip(point_b.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();

    distance_squared
}

unsafe extern "C" fn calc_centroid(
    objs: *const Pointer,
    clusters: *const c_int,
    num_objs: usize,
    cluster: c_int,
    centroid: Pointer,
) {
    let mut sum_by_dimension: Vec<f64> = vec![0.0; 3];
    let mut count = 0;

    for i in 0..num_objs {
        if *clusters.add(i) == cluster {
            let point = &*(objs.add(i) as *const Vec<f64>);
            for j in 0..3 {
                sum_by_dimension[j] += point[j];
            }
            count += 1;
        }
    }

    if count > 0 {
        for j in 0..3 {
            *(((centroid as *mut Vec<f64>).add(0)) as *mut f64).add(j) = sum_by_dimension[j] / count as f64;
        }
    }
}


pub fn cluster_with_kmeans(data: &Vec<Vec<f64>>, cluster_count: u32, max_iteration_count: u32) {
    let mut config = kmeans_config {
        distance_method: Some(calc_distance),
        centroid_method: Some(calc_centroid),
        objs: ptr::null_mut(),
        num_objs: data.len(),
        centers: ptr::null_mut(),
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
    for _ in 0..cluster_count {
        centers.push(vec![0.0f64; 3]);
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

    // Handle the result or do further processing as needed
    println!("Kmeans result: {:?}", result);
    println!("Kmeans centers: {:?}", centers);
    println!("Kmeans clusters: {:?}", clusters);
}

pub fn generate_random_data_points(num_objs: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::new(0.0, 1.0);

    (0..num_objs)
        .map(|_| {
            (0..3)
                .map(|_| distribution.sample(&mut rng))
                .collect::<Vec<f64>>()
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_clustering() {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0f64, 0.1f64, 0.0f64],
            vec![0.1f64, 0.1f64, 0.0f64],
            vec![0.0f64, 0.1f64, 0.1f64],
            vec![2.0f64, 2.1f64, 2.0f64],
            vec![2.1f64, 2.1f64, 2.0f64],
            vec![2.0f64, 2.1f64, 2.1f64],
        ];
        // let data = generate_random_data_points(10);
        println!("Data: {:?}", data);
        println!("Data Length: {:?}", data.len());
        cluster_with_kmeans(&data, 2 as u32, 100 as u32);
    }
}
