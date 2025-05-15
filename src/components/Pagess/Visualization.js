import React from 'react';

const Visualization = () => {
  return (
    <div className="text-center p-6 bg-gray-100 min-h-screen py-14">
      <h1 className="text-4xl font-bold text-gray-800 mb-4 text-primary">Data Visualization and Insights</h1>
      <p className="text-gray-600 mb-8">Below are the charts showing data for multiple parameters obtained from sensors</p>

      {/* Grid with two charts per row on larger screens */}
      <div className="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-6">
        {/* ThingSpeak Chart 1 */}
        <div className="bg-white p-4 rounded-lg shadow-md w-full">
          <iframe
            className="w-full h-[300px] rounded-lg"
            src="https://thingspeak.com/channels/2693814/charts/2?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=200&type=line"
            title="ThingSpeak Chart 1"
          ></iframe>
        </div>

        {/* ThingSpeak Chart 2 */}
        <div className="bg-white p-4 rounded-lg shadow-md w-full">
          <iframe
            className="w-full h-[300px] rounded-lg"
            src="https://thingspeak.com/channels/2693814/charts/3?bgcolor=%23ffffff&color=%23006b6b&dynamic=true&results=200&type=line"
            title="ThingSpeak Chart 2"
          ></iframe>
        </div>

        {/* ThingSpeak Chart 3 */}
        <div className="bg-white p-4 rounded-lg shadow-md w-full">
          <iframe
            className="w-full h-[300px] rounded-lg"
            src="https://thingspeak.com/channels/2693814/charts/4?bgcolor=%23ffffff&color=%230066cc&dynamic=true&results=200&type=line"
            title="ThingSpeak Chart 3"
          ></iframe>
        </div>

        {/* ThingSpeak Chart 4 */}
        <div className="bg-white p-4 rounded-lg shadow-md w-full">
          <iframe
            className="w-full h-[300px] rounded-lg"
            src="https://thingspeak.com/channels/2693814/charts/5?bgcolor=%23ffffff&color=%23d6d600&dynamic=true&results=200&type=line"
            title="ThingSpeak Chart 4"
          ></iframe>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md w-full">
          <iframe
            className="w-full h-[300px] rounded-lg"
            src="https://thingspeak.com/channels/2693814/charts/6?bgcolor=%23ffffff&color=%237f8c8d&dynamic=true&results=200&type=line"
            title="ThingSpeak Chart 4"
          ></iframe>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md w-full">
          <iframe
            className="w-full h-[300px] rounded-lg"
            src="https://thingspeak.com/channels/2693814/charts/7?bgcolor=%23ffffff&color=%239b59b6&dynamic=true&results=200&type=line"
            title="ThingSpeak Chart 4"
          ></iframe>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md w-full">
          <iframe
            className="w-full h-[300px] rounded-lg"
            src="https://thingspeak.com/channels/2693814/charts/8?bgcolor=%23ffffff&color=%23f39c12&dynamic=true&results=200&type=line"
            title="ThingSpeak Chart 4"
          ></iframe>
        </div>
      </div>
    </div>
  );
};

export default Visualization;
