# Not Supported Operators

## rope

```c++
if (((D / 2 % 32) != 0) || (n_rot / 2 <= 32) || (mode != 2)) {
    throw std::runtime_error("ROPE Parameter not Supportted");
}
```

throw error，the D param is 32, so (D / 2 % 32)==16

## flash_attn_ext

there is no mask parameter, only Q,K,V