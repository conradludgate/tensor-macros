pub trait Tensor {
    const SIZE: usize;
    const NDIM: usize;

    fn dims(&self) -> Vec<usize>;
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

pub enum TensorError {
    WrongSize,
}

#[macro_export]
macro_rules! make_tensor {
	($name:ident $($dim:literal) * ) => {

		/// A Tensor of size $($dim)x*
		///
		/// Generated by tensor-macros
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
	};
}

#[macro_export]
macro_rules! tensor {
	($name:ident $dim:literal) => {
		make_tensor!($name $dim);

		impl<T> Vector for $name<T> {
			const COLS = $dim;
		}
	};

	($name:ident row $dim:literal) => {
		make_tensor!($name $dim);

		impl<T> RowVector for $name<T> {
			const ROWS = $dim;
		}
	};

	($name:ident $dim1:literal $dim2:literal) => {
		make_tensor!($name $dim1 $dim2);

		impl<T> Matrix for $name<T> {
			const ROWS = $dim1;
			const COLS = $dim2;
		}
	};

	($name:ident $($dim:literal) * ) => (
		make_tensor!($name $($dim) *);
	)
}
