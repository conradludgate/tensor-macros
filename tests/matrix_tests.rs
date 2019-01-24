#![feature(try_from)]

#[macro_use]
extern crate tensor_macros;
use std::convert::TryFrom;
use tensor_macros::Tensor;
use tensor_macros::TensorError;

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
}
