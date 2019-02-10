#![feature(try_from)]

#[macro_use]
extern crate tensor_macros;
use tensor_macros::traits::*;

use std::convert::TryFrom;

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

dot!(T243: 2 x 4 x 3 * M43: 4 x 3 x 1 => V2: 2 x 1);

#[test]
fn dot_test() {
    let l = T243::<f64>([
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
    ]);
    let r = M43([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    assert_eq!(l * r, V2([506.0, 1298.0]));
}

tensor!(T2345: 2 x 3 x 4 x 5);

#[test]
fn debug() {
    let t = T2345::try_from((0u8..120).collect::<Vec<u8>>()).unwrap();
    let output = "0\t1\t2\t3\t4\t
5\t6\t7\t8\t9\t
10\t11\t12\t13\t14\t
15\t16\t17\t18\t19\t

20\t21\t22\t23\t24\t
25\t26\t27\t28\t29\t
30\t31\t32\t33\t34\t
35\t36\t37\t38\t39\t

40\t41\t42\t43\t44\t
45\t46\t47\t48\t49\t
50\t51\t52\t53\t54\t
55\t56\t57\t58\t59\t


60\t61\t62\t63\t64\t
65\t66\t67\t68\t69\t
70\t71\t72\t73\t74\t
75\t76\t77\t78\t79\t

80\t81\t82\t83\t84\t
85\t86\t87\t88\t89\t
90\t91\t92\t93\t94\t
95\t96\t97\t98\t99\t

100\t101\t102\t103\t104\t
105\t106\t107\t108\t109\t
110\t111\t112\t113\t114\t
115\t116\t117\t118\t119\t


";

    assert_eq!(format!("{:?}", t), output);
}

#[test]
fn cwise() {
    let l = T243::try_from((0u64..24).collect::<Vec<u64>>()).unwrap();
    let r = T243::from(5);

    assert_eq!(
        l.cwise_mul(l.cwise_mul(r)),
        T243::try_from((0u64..24).map(|x| x * x * 5).collect::<Vec<u64>>()).unwrap()
    );
}
