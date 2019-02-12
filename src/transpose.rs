#[macro_export]
macro_rules! make_transpose_index_fn {
	($name:ident; $dim:literal) => {
		impl<T: tensor_macros::traits::TensorTrait> std::ops::Index<usize> for $name<T> {
			type Output = T;

			fn index(&self, i: usize) -> &Self::Output {
				&self.0[i]
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::ops::IndexMut<usize> for $name<T> {
			fn index_mut(&mut self, i: usize) -> &mut T {
				&mut self.0[i]
			}
		}
	};

	($name:ident; $($dims:literal),+) => {
		impl<T: tensor_macros::traits::TensorTrait> std::ops::Index<usize> for $name<T> {
			type Output = T;

			fn index(&self, i: usize) -> &Self::Output {
				&self.0[i]
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::ops::IndexMut<usize> for $name<T> {
			fn index_mut(&mut self, i: usize) -> &mut T {
				&mut self.0[i]
			}
		}

		make_transpose_index_fn!($name; $($dims),*;;;);
	};

	($name:ident; $dim:literal $(,$dims:literal)*; $($i:ident),*; $($t:ty),*; $($dims_bk:literal),*) => {
		make_transpose_index_fn!($name; $($dims),*; $($i,)* i; $($t,)* usize; $($dims_bk,)* $dim);
	};
	($name:ident; ; $($i:ident),*; $($t:ty),*; $($dims:literal),*) => {
		impl<T: tensor_macros::traits::TensorTrait> std::ops::Index<( $($t),* )> for $name<T> {
			type Output = T;

			fn index(&self, ( $($i),* ): ( $($t),* )) -> &Self::Output {
				&self.0[
					make_transpose_index_val!($($dims),*; $($i),*;)
				]
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::ops::IndexMut<( $($t),* )> for $name<T> {
			fn index_mut(&mut self, ( $($i),* ): ( $($t),* )) -> &mut T {
				&mut self.0[
					make_transpose_index_val!($($dims),*; $($i),*;)
				]
			}
		}
	};
}

#[macro_export]
macro_rules! make_transpose_index_val {
	($($dims:literal),*; $i:expr $(,$is:expr)* ; $($js:expr),*) => (
		make_transpose_index_val!($($dims),*; $($is),*; $i $(,$js)*)
	);
	($($dims:literal),*; ; $($js:expr),*) => (
		make_transpose_index_val!(~$($dims),*; $($js),*)
	);
	(~$dim:literal $(,$dims:literal)*; $i:expr $(,$is:expr)* ) => (
		$i * mul!($($dims),*) + make_transpose_index_val!(~$($dims),*; $($is),*)
	);
	(~;) => (0)
}

#[macro_export]
/// Creates a tensor transpose type and function. Transposing doesn't cost anything
///
/// # Example
///
/// ```rust
/// #![feature(try_from)]
///
/// #[macro_use]
/// use tensor_macros::*;
/// use tensor_macros::traits::*;
///
/// tensor!(M23: 2 x 3);
/// tensor!(M33: 3 x 3);
/// transpose!(M23: 2 x 3 => M32);
/// dot!(M32: 3 x 2 * M23: 2 x 3 => M33: 3 x 3);
///
/// let t = M23([0, 1, 2, 3, 4, 5]);
/// let u = t.transpose();
/// let v = M33([9, 12, 15, 12, 17, 22, 15, 22, 29]);
/// assert_eq!(u * t, v);
/// ```
macro_rules! transpose {
    ($from:ident: $($dim:literal)x+ => $to:ident) => {
    	transpose!(~ $from, $to; $($dim),*;;);
    };
    (~ $from:ident, $to:ident; $d:literal $(,$dims:literal)*; $($fd:literal),*; $($td:literal),*) => {
    	transpose!(~ $from, $to; $($dims),*; $($fd,)* $d; $d $(,$td)*);
    };
    (~ $from:ident, $to:ident;; $($fd:literal),*; $($td:literal),*) => {
    	transpose!($from: $($fd)x* => $to: $($td)x*);
    };
    ($from:ident: $($from_dim:literal)x+ => $to:ident: $($to_dim:literal)x+) => {
    	// #[derive(Copy, Clone, PartialEq, Default)]
        pub struct $to<T: tensor_macros::traits::TensorTrait>($from<T>);

        impl<T: tensor_macros::traits::TensorTrait> $to<T> {
			#[allow(dead_code)]
			fn new() -> Self {
				Default::default()
			}
		}

        impl<T> tensor_macros::traits::TensorTranspose<$to<T>, T> for $from<T>
        where
            T: tensor_macros::traits::TensorTrait,
        {
            fn transpose(self) -> $to<T> {
                $to(self)
            }
        }

        impl<T> tensor_macros::traits::TensorTranspose<$from<T>, T> for $to<T>
        where
            T: tensor_macros::traits::TensorTrait,
        {
            fn transpose(self) -> $from<T> {
                self.0
            }
        }

        impl<T> tensor_macros::traits::Tensor for $to<T>
        	where T: tensor_macros::traits::TensorTrait,
        {
			type Value = T;

			const SIZE: usize = <$from<T> as tensor_macros::traits::Tensor>::SIZE;
			const NDIM: usize = <$from<T> as tensor_macros::traits::Tensor>::NDIM;

			fn dims() -> Vec<usize> {
				vec!($($to_dim),*)
			}

			fn get_dims(&self) -> Vec<usize> {
				Self::dims()
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> Copy for $to<T> { }

		impl<T: tensor_macros::traits::TensorTrait> Clone for $to<T> {
			fn clone(&self)	-> Self {
				$to(self.0)
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> PartialEq for $to<T> {
			fn eq(&self, other: &Self) -> bool {
				self.0 == other.0
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> Default for $to<T> {
			fn default() -> Self {
				$to($from::default())
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::convert::TryFrom<&[T]> for $to<T> {
			type Error = tensor_macros::traits::TensorError;

			fn try_from(v: &[T]) -> Result<Self, Self::Error> {
				Ok($to($from::try_from(v)?))
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::convert::TryFrom<Vec<T>> for $to<T> {
			type Error = tensor_macros::traits::TensorError;

			fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
				Ok($to($from::try_from(v)?))
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::convert::From<T> for $to<T> {
			fn from(t: T) -> Self {
				$to($from::from(t))
			}
		}

		make_transpose_index_fn!($to; $($from_dim),*);

		impl<T: tensor_macros::traits::TensorTrait> std::fmt::Debug for $to<T>  {
			fn fmt(&self, f:  &mut std::fmt::Formatter) -> std::fmt::Result {
				debug_tensor!(f, self; $($to_dim),*;);

				Ok(())
			}
		}

		impl<T, U, V> std::ops::Add<$to<U>> for $to<T>
			where Self: tensor_macros::traits::Tensor<Value=T>,
			T: tensor_macros::traits::TensorTrait + std::ops::Add<U, Output=V>,
			U: tensor_macros::traits::TensorTrait,
			V: tensor_macros::traits::TensorTrait,
		{
			type Output = $to<V>;

			fn add(self, other: $to<U>) -> Self::Output {
				$to(self.0 + other.0)
		    }
		}

		impl<T, U> std::ops::AddAssign<$to<U>> for $to<T>
			where T: tensor_macros::traits::TensorTrait + std::ops::AddAssign<U>,
			U: tensor_macros::traits::TensorTrait,
		{
			fn add_assign(&mut self, other: $to<U>) {
				for i in 0..mul!($($to_dim),*) {
					self[i] += other[i];
				}
		    }
		}

		impl<T, U, V> tensor_macros::traits::CwiseMul<$to<U>> for $to<T>
			where T: tensor_macros::traits::TensorTrait + std::ops::Mul<U, Output=V>,
			U: tensor_macros::traits::TensorTrait,
			V: tensor_macros::traits::TensorTrait,
		{
			type Output = $to<V>;

			fn cwise_mul(self, other: $to<U>) -> Self::Output {
				$to(self.0.cwise_mul(other.0))
			}
		}

		impl<T, U> tensor_macros::traits::CwiseMulAssign<$to<U>> for $to<T>
			where T: tensor_macros::traits::TensorTrait + std::ops::MulAssign<U>,
			U: tensor_macros::traits::TensorTrait,
		{
			fn cwise_mul_assign(&mut self, other: $to<U>) {
				for i in 0..mul!($($to_dim),*) {
					self[i] *= other[i];
				}
			}
		}

		impl<T, U> std::ops::MulAssign<U> for $to<T>
			where T: tensor_macros::traits::TensorTrait + std::ops::MulAssign<U>,
			U: Clone,
		{
			fn mul_assign(&mut self, other: U) {
				for i in 0..mul!($($to_dim),*) {
					self[i] *= other.clone();
				}
			}
		}
    };
}
