#[macro_use]
extern crate tensor_macros;
use tensor_macros::Tensor;

tensor!(T324 3 x 2 x 4);

#[test]
fn types_to_string() {
    let t324: T324<f64> = Default::default();
    assert_eq!(t324.dims(), vec!(3, 2, 4));
}
