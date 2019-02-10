#![feature(try_from)]
#![feature(test)]

extern crate test;

// #[macro_use]
// pub mod functional;

pub mod defaults;
pub mod traits;

#[macro_use]
pub mod debug;

#[macro_use]
pub mod index;

#[macro_use]
pub mod tensor;

#[macro_use]
pub mod dot;

#[cfg(test)]
mod tests {
    use crate as tensor_macros;
    tensor!(T243: 2 x 4 x 3);
    tensor!(M43: 4 x 3 x 1);
    tensor!(V2: 2 x 1);

    dot!(T243: 2 x 4 x 3 * M43: 4 x 3 x 1 => V2: 2 x 1);

    use test::{black_box, Bencher};
    #[bench]
    fn tensor_macros_dot_bench(b: &mut Bencher) {
        let l = T243([
            0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        ]);
        let r = M43([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

        assert_eq!(l * r, V2([506.0, 1298.0]));

        let mut o = V2::<f64>::new();

        b.iter(|| {
            for _i in 0..100 {
                o += black_box(l * r);
            }

            assert_eq!(o, V2([100.0 * 506.0, 100.0 * 1298.0]));
            o = V2::<f64>::new()
        });

        assert_eq!(o, V2([0.0, 0.0]));
    }
}
