# tensor

Compile time optimised tensor library

## Examples

```rust
#[macro_use]
extern crate tensor-macros;

tensor!(T324 3x2x4)
matrix!(M24 2x4)   // Alias for tensor!(M24 2x4) but only allows two dimensions
matrix!(M22 2x2)
vector!(V3 3)      // Alias for tensor!(V3 3) but only allows one dimension

dot!(T324 * M24 -> V3)
dot!(M22 * M24 -> M24)

// Will not compile
// dot!(M24 * M22 -> M22)

fn main() {
    let m22: M22 = vec![
        1, 2, 
        3, 4,
    ];
        
    let m24: M24 = vec![
        1, 2, 3, 4,
        5, 6, 7, 8,
    ];
    
    assert_eq!(m22 * m24, vec![
        11, 14, 17, 20,
        23, 30, 37, 44
    ]);
}

```