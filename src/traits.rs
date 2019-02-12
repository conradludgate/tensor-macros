use std::fmt::Debug;
use std::ops::Index;
use std::ops::IndexMut;

pub trait TensorTrait:
    PartialEq
    + Debug
    + Default
    + std::ops::Add
    + std::ops::AddAssign
    + std::ops::Mul
    + std::ops::MulAssign
    + Copy
    + Clone
{
}

impl<T> TensorTrait for T where
    T: PartialEq
        + Debug
        + Default
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::Mul
        + std::ops::MulAssign
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

pub trait CwiseMul<Rhs: Tensor> {
    type Output: Tensor;
    fn cwise_mul(self, other: Rhs) -> Self::Output;
}

pub trait CwiseMulAssign<Rhs: Tensor> {
    fn cwise_mul_assign(&mut self, other: Rhs);
}

// pub struct TensorTranspose<T: TensorTrait, TT: Tensor<Value = T>>(TT);

pub trait TensorTranspose<T, TT>: Tensor<Value = TT>
where
    T: Tensor<Value = TT>,
    // + std::ops::Add
    // + std::ops::AddAssign
    // + std::ops::Mul<TT>
    // + std::ops::MulAssign<TT>,
    TT: TensorTrait,
{
    fn transpose(self) -> T;
}

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
