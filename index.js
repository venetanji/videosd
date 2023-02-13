const express = require('express')
const app = express()
const port = 3000
app.use(express.static('public'))

app.listen(port, '0.0.0.0', () => {
  console.log(`Example app listening on port ${port}`)
})