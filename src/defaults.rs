use crate as tensor_macros;
use crate::*;

tensor!(V2: 2 x 1);
tensor!(V3: 3 x 1);
tensor!(V4: 4 x 1);

transpose!(V2: 2 x 1 => RowV2);
transpose!(V3: 3 x 1 => RowV3);
transpose!(V4: 4 x 1 => RowV4);

tensor!(M22: 2 x 2);
tensor!(M23: 2 x 3);
tensor!(M24: 2 x 4);
tensor!(M33: 3 x 3);
tensor!(M34: 3 x 4);
tensor!(M44: 4 x 4);

transpose!(M22: 2 x 2 => M22T);
transpose!(M23: 2 x 3 => M32);
transpose!(M24: 2 x 4 => M42);
transpose!(M33: 3 x 3 => M33T);
transpose!(M34: 3 x 4 => M43);
transpose!(M44: 4 x 4 => M44T);

dot!(M22: 2 x 2 * V2: 2 x 1 => V2: 2 x 1);
dot!(M32: 3 x 2 * V2: 2 x 1 => V3: 3 x 1);
dot!(M42: 4 x 2 * V2: 2 x 1 => V4: 4 x 1);

dot!(M23: 2 x 3 * V3: 3 x 1 => V2: 2 x 1);
dot!(M33: 3 x 3 * V3: 3 x 1 => V3: 3 x 1);
dot!(M43: 4 x 3 * V3: 3 x 1 => V4: 4 x 1);

dot!(M24: 2 x 4 * V4: 4 x 1 => V2: 2 x 1);
dot!(M34: 3 x 4 * V4: 4 x 1 => V3: 3 x 1);
dot!(M44: 4 x 4 * V4: 4 x 1 => V4: 4 x 1);

dot!(M22: 2 x 2 * M22: 2 x 2 => M22: 2 x 2);
dot!(M23: 2 x 3 * M32: 3 x 2 => M22: 2 x 2);
dot!(M24: 2 x 4 * M42: 4 x 2 => M22: 2 x 2);

dot!(M32: 3 x 2 * M22: 2 x 2 => M32: 3 x 2);
dot!(M33: 3 x 3 * M32: 3 x 2 => M32: 3 x 2);
dot!(M34: 3 x 4 * M42: 4 x 2 => M32: 3 x 2);

dot!(M42: 4 x 2 * M22: 2 x 2 => M32: 4 x 2);
dot!(M43: 4 x 3 * M32: 3 x 2 => M32: 4 x 2);
dot!(M44: 4 x 4 * M42: 4 x 2 => M32: 4 x 2);

dot!(M22: 2 x 2 * M22T: 2 x 2 => M22: 2 x 2);
dot!(M22T: 2 x 2 * M22: 2 x 2 => M22: 2 x 2);

dot!(M22: 2 x 2 * M23: 2 x 3 => M23: 2 x 3);
dot!(M23: 2 x 3 * M33: 3 x 3 => M23: 2 x 3);
dot!(M24: 2 x 4 * M43: 4 x 3 => M23: 2 x 3);

dot!(M32: 3 x 2 * M23: 2 x 3 => M33: 3 x 3);
dot!(M33: 3 x 3 * M33: 3 x 3 => M33: 3 x 3);
dot!(M34: 3 x 4 * M43: 4 x 3 => M33: 3 x 3);

dot!(M42: 4 x 2 * M23: 2 x 3 => M33: 4 x 3);
dot!(M43: 4 x 3 * M33: 3 x 3 => M33: 4 x 3);
dot!(M44: 4 x 4 * M43: 4 x 3 => M33: 4 x 3);

dot!(M33: 3 x 3 * M33T: 3 x 3 => M33: 3 x 3);
dot!(M33T: 3 x 3 * M33: 3 x 3 => M33: 3 x 3);

dot!(M22: 2 x 2 * M24: 2 x 4 => M24: 2 x 4);
dot!(M23: 2 x 3 * M34: 3 x 4 => M24: 2 x 4);
dot!(M24: 2 x 4 * M44: 4 x 4 => M24: 2 x 4);

dot!(M32: 3 x 2 * M24: 2 x 4 => M34: 3 x 4);
dot!(M33: 3 x 3 * M34: 3 x 4 => M34: 3 x 4);
dot!(M34: 3 x 4 * M44: 4 x 4 => M34: 3 x 4);

dot!(M42: 4 x 2 * M24: 2 x 4 => M34: 4 x 4);
dot!(M43: 4 x 3 * M34: 3 x 4 => M34: 4 x 4);
dot!(M44: 4 x 4 * M44: 4 x 4 => M34: 4 x 4);

dot!(M44: 4 x 4 * M44T: 4 x 4 => M44: 4 x 4);
dot!(M44T: 4 x 4 * M44: 4 x 4 => M44: 4 x 4);
