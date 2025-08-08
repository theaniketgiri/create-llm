module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/test/test.js'],
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/templates/**/*.js'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html']
}; 