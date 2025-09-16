def remap(
    input: float,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
) -> float:
    """
    Remaps a value from the input to the output range.
    """
    input_range_size = input_range[1] - input_range[0]
    output_range_size = output_range[1] - output_range[0]
    output_value = (
        ((input - input_range[0]) * output_range_size) / input_range_size
    ) + output_range[0]

    return output_value
