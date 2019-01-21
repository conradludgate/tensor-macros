pub trait Tensor {
    const SIZE: usize;
    const NDIM: usize;

    fn dims(&self) -> Vec<usize>;
}

pub enum TensorError {
    WrongSize,
}

#[macro_export]
macro_rules! tensor {
	($name:ident $($dim:literal) * ) => {

		#[derive(Debug,Default)]
		pub struct $name<T> (
			[T; 1 $( * $dim )*]
		);

		impl<T> Tensor for $name<T> {
			const SIZE: usize = 1 $( * $dim )*;
			const NDIM: usize = 0 $( + $dim/$dim )*;

			fn dims(&self) -> Vec<usize> {
				vec!($($dim),*)
			}
		}
	}
}
