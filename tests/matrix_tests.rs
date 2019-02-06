#![feature(try_from)]
// #![feature(test)]

#[macro_use]
extern crate tensor_macros;
// extern crate test;
use std::convert::TryFrom;
use tensor_macros::tensor::{Matrix, Tensor, TensorError};

tensor!(T324: 3 x 2 x 4);

#[test]
fn dims() {
    assert_eq!(T324::<u8>::dims(), vec!(3, 2, 4));

    let t324: T324<f64> = Default::default();
    assert_eq!(t324.get_dims(), vec!(3, 2, 4));
}

#[test]
fn try_from_vec() {
    let t324 = T324::<u8>::try_from(vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    ]);

    let exp = T324([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    ]);

    assert_eq!(t324, Ok(exp));
}

#[test]
fn index() {
    let t324 = T324::<u8>::try_from(vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    ])
    .unwrap();

    assert_eq!(t324[(0, 0, 0)], 0);
    assert_eq!(t324[(1, 1, 1)], 13);
    assert_eq!(t324[(2, 1, 3)], 23);
    assert_eq!(t324[15], 15);
}

tensor!(T243: 2 x 4 x 3);
tensor!(M43: 4 x 3 x 1);
tensor!(V2: 2 x 1);

dot!(T243: 2 x 4 x 3 * M43: 4 x 3 x 1 -> V2: 2 x 1);

#[test]
fn dot_test() {
    let l = T243([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    ]);
    let r = M43([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(l * r, V2([121, 253]));
}

// use test::Bencher;
// #[bench]
// fn tensor_macros_dot_bench(b: &mut Bencher) {
//     b.iter(|| {
//         let l = T243([
//             0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
//             15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
//         ]);
//         let r = M43([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
//         l * r
//     });
// }

// extern crate numeric;

// #[bench]
// fn numeric_dot_bench(b: &mut Bencher) {
//     b.iter(|| {
//         let l = numeric::Tensor::new(vec![
//             0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
//             15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
//         ])
//         .reshape(&[2, 4, 3]);

//         let r = numeric::Tensor::new(vec![
//             0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
//         ])
//         .reshape(&[4, 3]);

//         l * r
//     });
// }
