use std::fmt::Debug;
use std::ops::Index;
use std::ops::IndexMut;

pub trait TensorTrait:
    PartialEq + Debug + Default + std::ops::Add + std::ops::AddAssign + std::ops::Mul + Copy + Clone
{
}

impl<T> TensorTrait for T where
    T: PartialEq
        + Debug
        + Default
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::Mul
        + Copy
        + Clone
{
}

pub trait Tensor:
    Index<usize> + IndexMut<usize> + PartialEq + Debug + Default + Copy + Clone
{
    type Value: TensorTrait;

    const SIZE: usize;
    const NDIM: usize;

    fn dims() -> Vec<usize>;
    fn get_dims(&self) -> Vec<usize>;
    // fn transpose(self) -> TensorTranspose<Self>;
}

pub struct TensorTranspose<T: TensorTrait, TT: Tensor<Value = T>>(TT);

pub trait Matrix {
    const ROWS: usize;
    const COLS: usize;
}

pub trait Vector {
    const COLS: usize;
}

pub trait RowVector {
    const ROWS: usize;
}

#[derive(Debug, PartialEq)]
pub enum TensorError {
    Size,
}
