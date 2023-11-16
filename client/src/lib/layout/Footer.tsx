import { Flex, Link, Text, Image, Spacer, VStack, Divider, Box, Center } from '@chakra-ui/react';

const Footer = () => {
  return (
    <Box w='full'>

    <Divider orientation='horizontal' colorScheme='dark'   />
    <Flex margin={'auto'} maxW={['full','300']} px={4} as="footer" fontSize={'2xs'} color={'gray.400'} fontWeight={'light'} direction={'row'} alignItems='center' justifyContent={"space-between"} py={1} textAlign={'center'}>
      <Text>
        Developed by<br/>
        <Link href="https://github.com/venetanji/videosd" mb={1} isExternal rel="noopener noreferrer">
          @venetanji
        </Link>
      </Text>
      <Link href="https://giovannilion.link" mb={1} isExternal rel="noopener noreferrer">
        <Image src="/pxl_gio.png" alt="Giovanni Lion"  h={7} />
      </Link>
      <Divider orientation='vertical' h={9} colorScheme='dark' />
      <Text>
        Sponsored by <br/>
        <Link href="https://redmond.ai"  isExternal rel="noopener noreferrer">
          redmond.ai
        </Link>
      </Text>
      
      <Link href="https://redmond.ai"  isExternal rel="noopener noreferrer">
        <Image src="/redmond.png" alt="Redmond AI" height={7} />
      </Link>
    </Flex>
    </Box>
  );
};

export default Footer;
