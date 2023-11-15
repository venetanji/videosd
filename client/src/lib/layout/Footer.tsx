import { Flex, Link, Text, Image, Spacer } from '@chakra-ui/react';

const Footer = () => {
  return (
    <Flex as="footer" width="full" direction={'row'} alignItems='center' justifyContent="center" p={2} mt={[3,2,4]}>

      <Link href="https://giovannilion.link" isExternal rel="noopener noreferrer">
        <Image src="/pxl_gio.png" alt="Giovanni Lion" height="5vh" />
      </Link>
      <Text mx={1} fontSize={30}> + </Text>
      <Link href="https://redmond.ai" isExternal rel="noopener noreferrer">
        <Image src="https://redmond.ai/wp-content/uploads/2023/02/logo.png" alt="Redmond AI" height="4vh" mt={2} />
      </Link>

    </Flex>
  );
};

export default Footer;
