#![feature(try_from)]

pub trait Tensor {
    const SIZE: usize;
    const NDIM: usize;

    fn dims() -> Vec<usize>;
    fn get_dims(&self) -> Vec<usize>;
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

#[macro_export]
macro_rules! mul {
	() => (1);
	($head:expr) => ($head);
	($head:expr, $($tail:expr),+) => ($head * mul!($($tail),*));
}

#[macro_export]
macro_rules! sum {
	() => (0);
	($head:expr) => (1);
	($head:expr, $($tail:expr),+) => (1 + sum!($($tail),*));
}

#[macro_export]
/// Generate consecutive pairs from a list of inputs
///
/// Used internally by other macros.
/// Takes the first two values,
/// adds them both to the end and discards the first one
///
/// # Example
///
/// ```rust
/// use tensor_macros::pairs;
/// let v = pairs!(1, 2, 3, 4, 5);
/// assert_eq!(v, ((1, 2), (2, 3), (3, 4), (4, 5)));
/// ```
macro_rules! pairs {
	($_1:literal, $_2:literal $(,$tail:literal)*) => {
		pairs!($_2 $(,$tail)*; $_1, $_2);
	};

	($_1:literal, $_2:literal $(,$tail:literal)*; $($x:literal, $y:literal),*) => {
		pairs!($_2 $(,$tail)*; $($x, $y),*, $_1, $_2);
	};

	($_1:literal; $($x:literal, $y:literal),*) => {
		($(($x, $y)),*)
	};
}

/// Generates a tensor type
///
/// Use [`tensor!`] instead, it uses this macro and more
///
/// [`tensor!`]: macro.tensor.html
#[macro_export]
macro_rules! make_tensor {
	($name:ident $($dim:literal)x+ ) => {

		/// A Tensor of size $($dim)x*
		///
		/// Generated by [tensor-macros](https://github.com/conradludgate/tensor-macros)
		pub struct $name<T> ([T; mul!($($dim),*)]);

		impl<T> Tensor for $name<T> {
			const SIZE: usize = mul!($($dim),*);
			const NDIM: usize = sum!($($dim),*);

			fn dims() -> Vec<usize> {
				vec!($($dim),*)
			}

			fn get_dims(&self) -> Vec<usize> {
				Self::dims()
			}
		}

		impl<T: PartialEq> PartialEq for $name<T> {
			fn eq(&self, other: &Self) -> bool {
				for (p, q) in self.0.iter().zip(other.0.iter()) {
					if p != q {
						return false;
					}
				}

				true
			}
		}

		impl<T: std::fmt::Debug> std::fmt::Debug for $name<T>  {
			fn fmt(&self, f:  &mut std::fmt::Formatter) -> std::fmt::Result {
				for i in self.0.iter() {
					write!(f, "{:?} ", i)?
				}

				Ok(())
			}
		}

		impl<T: Default + Copy> Default for $name<T> {
			fn default() -> Self {
				$name::<T>([Default::default(); mul!($($dim),*)])
			}
		}

		impl<T: Default + Copy> std::convert::TryFrom<&[T]> for $name<T> {
			type Error = TensorError;

			fn try_from(v: &[T]) -> Result<Self, Self::Error> {
				if v.len() < mul!($($dim),*) {
					Err(TensorError::Size)
				} else {
					let mut a: [T; mul!($($dim),*)] = [Default::default(); mul!($($dim),*)];
					a.copy_from_slice(&v[..mul!($($dim),*)]);
					Ok($name::<T>(a))
				}
			}
		}

		impl<T: Default + Copy> std::convert::TryFrom<Vec<T>> for $name<T> {
			type Error = TensorError;

			fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
				if v.len() < mul!($($dim),*) {
					Err(TensorError::Size)
				} else {
					let mut a: [T; mul!($($dim),*)] = [Default::default(); mul!($($dim),*)];
					a.copy_from_slice(&v[..mul!($($dim),*)]);
					Ok($name::<T>(a))
				}
			}
		}
	};
}

/// Generates a tensor type
///
/// Generates a type with the given name and dimensions (space seperated)
/// There's no upper limit on the amount of dimensions given
/// Matricies and vectors have special properties assigned to them
///
/// # Example
///
/// ```rust
/// #![feature(try_from)]
/// use tensor_macros::*;
///
/// tensor!(M23: 2 x 3);
///
/// assert_eq!(M23::<f64>::dims(), vec!(2, 3));
///
/// let m23: M23<f64> = Default::default();
/// assert_eq!(m23.get_dims(), vec!(2, 3));
/// ```
#[macro_export]
macro_rules! tensor {
	($name:ident: $dim:literal) => {
		make_tensor!($name $dim);

		impl<T> Vector for $name<T> {
			const COLS: usize = $dim;
		}
	};

	($name:ident: row $dim:literal) => {
		make_tensor!($name $dim);

		impl<T> RowVector for $name<T> {
			const ROWS: usize = $dim;
		}
	};

	($name:ident: $dim1:literal x $dim2:literal) => {
		make_tensor!($name $dim1 x $dim2);

		impl<T> Matrix for $name<T> {
			const ROWS: usize = $dim1;
			const COLS: usize = $dim2;
		}
	};

	($name:ident: $($dim:literal)x+ ) => (
		make_tensor!($name $($dim) x *);
	)
}

#[cfg(test)]
mod tests {
    use super::*;
    tensor!(T2345: 2 x 3 x 4 x 5);
    #[test]
    fn tensor_dims() {
        assert_eq!(T2345::<u8>::SIZE, 2 * 3 * 4 * 5);
        assert_eq!(T2345::<u8>::NDIM, 4);
    }

    tensor!(M23: 2 x 3);
    #[test]
    fn matrix_dims() {
        assert_eq!(M23::<u8>::ROWS, 2);
        assert_eq!(M23::<u8>::COLS, 3);
    }

    tensor!(V4: 4);
    #[test]
    fn col_vector_size() {
        assert_eq!(V4::<u8>::COLS, 4);
    }

    tensor!(V2: row 2);
    #[test]
    fn row_vector_size() {
        assert_eq!(V2::<u8>::ROWS, 2);
    }

    #[test]
    fn pairs() {
        let v = pairs!(1, 2, 3, 4, 5);
        assert_eq!(v, ((1, 2), (2, 3), (3, 4), (4, 5)));
    }
}
