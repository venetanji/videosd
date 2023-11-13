import { Flex, Link, Text } from '@chakra-ui/react';

const Footer = () => {
  return (
    <Flex as="footer"  width="full" justifyContent="center" pb={2} pt={[2,6]}>
      <Text fontSize={["3xs", "initial"]}>
        {new Date().getFullYear()} -{' '}
        <Link href="https://blendotron.art" isExternal rel="noopener noreferrer">
          blendotron.art
        </Link>
      </Text>
    </Flex>
  );
};

export default Footer;
