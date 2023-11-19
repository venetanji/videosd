import { Slider, SliderTrack, SliderFilledTrack, SliderThumb, Box, FormLabel, Flex} from "@chakra-ui/react";
import { useState } from "react";


interface SliderParameterProps {
    children: React.ReactNode;
    label: string;
    defaultValue: number;
    min: number;
    max: number;
    step: number;
    isDisabled: boolean;
    onChange: (value: number) => void;
}

export default function SliderParameter({
    label,
    defaultValue,
    min,
    max,
    step,
    isDisabled,
    onChange,
    children
}: SliderParameterProps) {
    const [value, setValue] = useState(defaultValue);

    const handleChange = (newValue: number) => {
        setValue(newValue);
        onChange(newValue);
    };

    return (
        <Box px={[1,0]}>
        <Flex justify="space-between" align={["left","right"]} mb={1} direction={["row","column"]}>
        <FormLabel py={1} pl={[3,0]} mb={0} order={[1,0]} fontSize={['xs','inherit']} minW={[28,0]}>{children}</FormLabel>
        <Slider order={[0,1]} aria-label={label} isDisabled={isDisabled} defaultValue={defaultValue} min={min} max={max} step={step} onChange={handleChange}>
            <SliderTrack>
                <SliderFilledTrack />
            </SliderTrack>
            <SliderThumb />
        </Slider>
        </Flex>
        </Box>
    );
}
